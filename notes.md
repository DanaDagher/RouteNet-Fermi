>>>>>> old notes  update them don't take it into consideration

this is old work...

# Step 1 — DONE: Understand Input/Output
Reverse-engineered delay_model.py completely. RouteNet-Fermi is a Graph Neural Network with message-passing between 3 entity types Path - Link - Queue anf Graph tensors
> Output: per-flow delay = queue_delay + transmission_delay

# Step 2 — DONE: Run predict.py on all traffic models
Saved predictions_delay_*.npy, labels_delay_*.npy, flow_counts_delay_*.npy for all 5 TMs:

Traffic Model	    Mean MAPE	Median MAPE
constant_bitrate    	4.29%	1.15%
onoff	                2.74%	0.99%
autocorrelated	        2.46%	0.93%
modulated	            5.26%	3.82%
all_multiplexed	        4.53%	2.48%

> Matches paper's reported MAPE. Model validated.

# Step 3 — SHAP Plan
The challenge: RouteNet-Fermi takes dict inputs with ragged/graph tensors — standard SHAP can't perturb them directly. The strategy:

Focus: 10 scalar path features per flow (the traffic, packets, etc.)
Method: shap.KernelExplainer — model-agnostic, works via perturbation
Wrapper: flatten one flow's 10 features, call full model with that flow's features replaced
Dataset: constant_bitrate (simplest, cleanest — good for baseline)
Samples: 40 flows (20 low-delay + 20 high-delay)
Background: median of 50 reference flows

# Used take(200) on test set because full set was too slow for SHAP iteration 

## After the execution of predict.py for delay prediction intraffic_models
  
Ran predict.py on 200 test samples × 5 traffic models = 54,400 flows each.
Saved predictions, labels, flow_counts as .npy files.

MAPE results:
| Traffic model      | Mean  | Median |
|--------------------|-------|--------|
| constant_bitrate   | 4.29% | 1.15%  |
| onoff              | 2.74% | 0.99%  |
| autocorrelated     | 2.46% | 0.93%  |
| modulated          | 5.26% | 3.82%  |
| all_multiplexed    | 4.53% | 2.48%  |

Model is performing as expected (matches paper's reported MAPE).
modulated is hardest, autocorrelated is easiest.


# Applying SHAP to delay(one choosen sub-dataset) traffic model only; extension to jitter/loss is easy
 
# Use constant_bitrate sub-dataset when Appling Graph SHAP on the pretrained model and Analyzing the imporatance scores of the inputs
# why ? it's the cleanest single-model baseline


# @RoutNet-Fermi > @traffic_models > delay > constant_bitrate 
