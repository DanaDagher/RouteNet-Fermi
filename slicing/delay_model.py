"""
RouteNet-Fermi delay model adapted to the Zenodo network-slicing dataset.

Single structural change vs the original ../delay_model.py:
  - path_embedding input: drops the 5 traffic-model params absent from this
    dataset (pkts_lambda_on, avg_t_off, avg_t_on, ar_a, sigma) and adds the
    slice features: slice_type one-hot(3) + delta.
    Input dim: 17 -> 16.
  - max_num_queues raised (up to ~35 per-slice WFQ queues per port here,
    vs 3 fixed queues in the scheduling dataset).
Message passing, state dims, iterations and readout are unchanged.

NOTE: z_score values below are rough placeholders good enough for a forward
pass smoke test. They MUST be recomputed on the training split before any
real training run.
"""
import tensorflow as tf


class RouteNet_Fermi(tf.keras.Model):
    def __init__(self):
        super(RouteNet_Fermi, self).__init__()

        self.max_num_models = 7

        self.num_policies = 4
        self.max_num_queues = 40

        self.iterations = 8
        self.path_state_dim = 32
        self.link_state_dim = 32
        self.queue_state_dim = 32

        self.z_score = {
            'traffic': [1972448.7068892422, 4626730.330623528],
            'packets': [1374.7619267090618, 1306.792683198166],
            'eq_lambda': [1736348.7016428192, 3764515.3689682074],
            'avg_pkts_lambda': [1374.6913089560148, 1306.7266367833008],
            'exp_max_factor': [10.0, 1.0],
            'delta': [0.5181214626391097, 0.28820038824433974],
            'capacity': [62342038.753159225, 50254619.11630739],
            'queue_size': [41266.18373390502, 102603.74447111612],
        }

        self.path_update = tf.keras.layers.GRUCell(self.path_state_dim)
        self.link_update = tf.keras.layers.GRUCell(self.link_state_dim)
        self.queue_update = tf.keras.layers.GRUCell(self.queue_state_dim)

        # 5 scalars + one_hot(model,7) + one_hot(slice_type,3) + delta = 16
        self.path_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=5 + self.max_num_models + 3 + 1),
            tf.keras.layers.Dense(self.path_state_dim, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(self.path_state_dim, activation=tf.keras.activations.relu)
        ])

        self.queue_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.max_num_queues + 2),
            tf.keras.layers.Dense(self.queue_state_dim, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(self.queue_state_dim, activation=tf.keras.activations.relu)
        ])

        self.link_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.num_policies + 1),
            tf.keras.layers.Dense(self.link_state_dim, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(self.link_state_dim, activation=tf.keras.activations.relu)
        ])

        self.readout_path = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, self.path_state_dim)),
            tf.keras.layers.Dense(int(self.link_state_dim / 2),
                                  activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(int(self.path_state_dim / 2),
                                  activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(1)
        ], name="PathReadout")

    @tf.function
    def call(self, inputs):
        traffic = inputs['traffic']
        packets = inputs['packets']
        length = inputs['length']
        model = inputs['model']
        eq_lambda = inputs['eq_lambda']
        avg_pkts_lambda = inputs['avg_pkts_lambda']
        exp_max_factor = inputs['exp_max_factor']
        slice_type = tf.one_hot(inputs['slice_type'], 3)
        delta = inputs['delta']

        capacity = inputs['capacity']
        policy = tf.one_hot(inputs['policy'], self.num_policies)

        queue_size = inputs['queue_size']
        priority = tf.one_hot(inputs['priority'], self.max_num_queues)
        weight = inputs['weight']

        queue_to_path = inputs['queue_to_path']
        link_to_path = inputs['link_to_path']
        path_to_link = inputs['path_to_link']
        path_to_queue = inputs['path_to_queue']
        queue_to_link = inputs['queue_to_link']

        path_gather_traffic = tf.gather(traffic, path_to_link[:, :, 0])
        load = tf.math.reduce_sum(path_gather_traffic, axis=1) / capacity

        pkt_size = traffic / packets

        path_state = self.path_embedding(tf.concat(
            [(traffic - self.z_score['traffic'][0]) / self.z_score['traffic'][1],
             (packets - self.z_score['packets'][0]) / self.z_score['packets'][1],
             tf.one_hot(model, self.max_num_models),
             (eq_lambda - self.z_score['eq_lambda'][0]) / self.z_score['eq_lambda'][1],
             (avg_pkts_lambda - self.z_score['avg_pkts_lambda'][0]) / self.z_score['avg_pkts_lambda'][1],
             (exp_max_factor - self.z_score['exp_max_factor'][0]) / self.z_score['exp_max_factor'][1],
             slice_type,
             (delta - self.z_score['delta'][0]) / self.z_score['delta'][1]], axis=1))

        link_state = self.link_embedding(tf.concat([load, policy], axis=1))

        queue_state = self.queue_embedding(
            tf.concat([(queue_size - self.z_score['queue_size'][0]) / self.z_score['queue_size'][1],
                       priority, weight], axis=1))

        for it in range(self.iterations):
            queue_gather = tf.gather(queue_state, queue_to_path)
            link_gather = tf.gather(link_state, link_to_path, name="LinkToPath")
            path_update_rnn = tf.keras.layers.RNN(self.path_update,
                                                  return_sequences=True,
                                                  return_state=True)
            previous_path_state = path_state

            path_state_sequence, path_state = path_update_rnn(tf.concat([queue_gather, link_gather], axis=2),
                                                              initial_state=path_state)

            path_state_sequence = tf.concat([tf.expand_dims(previous_path_state, 1), path_state_sequence], axis=1)

            path_gather = tf.gather_nd(path_state_sequence, path_to_queue)
            path_sum = tf.math.reduce_sum(path_gather, axis=1)
            queue_state, _ = self.queue_update(path_sum, [queue_state])

            queue_gather = tf.gather(queue_state, queue_to_link)

            link_gru_rnn = tf.keras.layers.RNN(self.link_update, return_sequences=False)
            link_state = link_gru_rnn(queue_gather, initial_state=link_state)

        capacity_gather = tf.gather(capacity, link_to_path)
        input_tensor = path_state_sequence[:, 1:].to_tensor()

        occupancy_gather = self.readout_path(input_tensor)
        length = tf.ensure_shape(length, [None])
        occupancy_gather = tf.RaggedTensor.from_tensor(occupancy_gather, lengths=length)

        queue_delay = tf.math.reduce_sum(occupancy_gather / capacity_gather,
                                         axis=1)
        trans_delay = pkt_size * tf.math.reduce_sum(1 / capacity_gather, axis=1)

        return queue_delay + trans_delay
