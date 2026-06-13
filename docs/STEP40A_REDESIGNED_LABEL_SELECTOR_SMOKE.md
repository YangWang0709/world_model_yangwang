# Step40A Redesigned-Label Selector Smoke

Step40A is a bounded smoke where a small student selector head learns from the redesigned answer sheet.
It is not final selector training, because it does not permit downstream use and does not claim context utility.

- label_variant: `global_spatial_prior_removed_residual`
- train_global_spatial_prior is built from each train split only.
- validation split samples are not used to construct that prior.
- the same train prior is then applied to train and validation samples to build targets.

Safety:
- bounded_selector_smoke_training_performed: `true`
- optimizer_scope: `redesigned_label_proxy_selector_head_only`
- final selector/current importance/world model/downstream training: `false`
- checkpoint/state_dict saved: `false`
- new data/model downloads: `false`
- action/language/future tokens used as selector input: `false`
