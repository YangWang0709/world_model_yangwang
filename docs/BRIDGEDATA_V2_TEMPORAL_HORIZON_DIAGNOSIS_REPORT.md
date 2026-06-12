# BridgeData V2 Temporal Horizon Diagnosis Report

- best_target_variant: `future_delta_last_minus_current`
- delta_target_increases_context_gain: `true`
- current_dominance_level: `low`
- full_context_noise_confirmed: `true`

```json
[
  {
    "current_only_val": 3.563509782155355,
    "full_context_noise_penalty_present": true,
    "full_context_val": 3.55788524945577,
    "interpretation": "proxy context has a modest positive held-out utility signal",
    "per_seed_val_table": [
      {
        "current_only": 3.7151153087615967,
        "current_only_val_best": 3.403984785079956,
        "full_context_reference": 3.722898244857788,
        "full_context_reference_val_best": 3.552140235900879,
        "proxy_importance_topk": 3.469925880432129,
        "proxy_importance_topk_val_best": 3.3931872844696045,
        "split_seed": 42
      },
      {
        "current_only": 3.8506276607513428,
        "current_only_val_best": 3.478692054748535,
        "full_context_reference": 3.86328387260437,
        "full_context_reference_val_best": 3.5793559551239014,
        "proxy_importance_topk": 3.459059476852417,
        "proxy_importance_topk_val_best": 3.4384968280792236,
        "split_seed": 123
      },
      {
        "current_only": 3.124786376953125,
        "current_only_val_best": 2.708583116531372,
        "full_context_reference": 3.0874736309051514,
        "full_context_reference_val_best": 2.9204795360565186,
        "proxy_importance_topk": 2.8519840240478516,
        "proxy_importance_topk_val_best": 2.851860761642456,
        "split_seed": 999
      }
    ],
    "proxy_gain_over_current": 0.08508090999575596,
    "proxy_topk_val": 3.2603231271107993,
    "target_variant": "future_mean_all4"
  },
  {
    "current_only_val": 4.27535096804301,
    "full_context_noise_penalty_present": true,
    "full_context_val": 3.994940678278605,
    "interpretation": "proxy context has a modest positive held-out utility signal",
    "per_seed_val_table": [
      {
        "current_only": 4.725236415863037,
        "current_only_val_best": 3.457245111465454,
        "full_context_reference": 3.912445306777954,
        "full_context_reference_val_best": 3.5125234127044678,
        "proxy_importance_topk": 3.771099090576172,
        "proxy_importance_topk_val_best": 3.5080959796905518,
        "split_seed": 42
      },
      {
        "current_only": 4.570921897888184,
        "current_only_val_best": 3.545755624771118,
        "full_context_reference": 4.672641277313232,
        "full_context_reference_val_best": 3.624652624130249,
        "proxy_importance_topk": 3.719696283340454,
        "proxy_importance_topk_val_best": 3.480206251144409,
        "split_seed": 123
      },
      {
        "current_only": 3.5298945903778076,
        "current_only_val_best": 2.6833362579345703,
        "full_context_reference": 3.399735450744629,
        "full_context_reference_val_best": 2.9020235538482666,
        "proxy_importance_topk": 3.1073038578033447,
        "proxy_importance_topk_val_best": 2.937424659729004,
        "split_seed": 999
      }
    ],
    "proxy_gain_over_current": 0.17370532377048162,
    "proxy_topk_val": 3.5326997439066568,
    "target_variant": "future_first"
  },
  {
    "current_only_val": 4.17805544535319,
    "full_context_noise_penalty_present": true,
    "full_context_val": 4.110239426294963,
    "interpretation": "full context remains noisy relative to selected context",
    "per_seed_val_table": [
      {
        "current_only": 4.272097110748291,
        "current_only_val_best": 3.8375895023345947,
        "full_context_reference": 4.450737476348877,
        "full_context_reference_val_best": 4.006524562835693,
        "proxy_importance_topk": 4.132214069366455,
        "proxy_importance_topk_val_best": 3.775988817214966,
        "split_seed": 42
      },
      {
        "current_only": 4.266550540924072,
        "current_only_val_best": 3.851430654525757,
        "full_context_reference": 4.322691440582275,
        "full_context_reference_val_best": 3.9838876724243164,
        "proxy_importance_topk": 4.135609149932861,
        "proxy_importance_topk_val_best": 3.8230268955230713,
        "split_seed": 123
      },
      {
        "current_only": 3.995518684387207,
        "current_only_val_best": 3.1144139766693115,
        "full_context_reference": 3.5572893619537354,
        "full_context_reference_val_best": 3.3386266231536865,
        "proxy_importance_topk": 3.360741376876831,
        "proxy_importance_topk_val_best": 3.2193870544433594,
        "split_seed": 999
      }
    ],
    "proxy_gain_over_current": 0.07225065597526789,
    "proxy_topk_val": 3.8761881987253823,
    "target_variant": "future_last"
  },
  {
    "current_only_val": 2.9830404917399087,
    "full_context_noise_penalty_present": false,
    "full_context_val": 1.4748462041219075,
    "interpretation": "delta prediction exposes context gain under existing tokens",
    "per_seed_val_table": [
      {
        "current_only": 2.5298871994018555,
        "current_only_val_best": 1.0294170379638672,
        "full_context_reference": 1.3898075819015503,
        "full_context_reference_val_best": 0.9529440999031067,
        "proxy_importance_topk": 1.6040986776351929,
        "proxy_importance_topk_val_best": 1.1096841096878052,
        "split_seed": 42
      },
      {
        "current_only": 2.368776798248291,
        "current_only_val_best": 1.0069514513015747,
        "full_context_reference": 1.4257159233093262,
        "full_context_reference_val_best": 0.959077775478363,
        "proxy_importance_topk": 1.505800724029541,
        "proxy_importance_topk_val_best": 1.0829542875289917,
        "split_seed": 123
      },
      {
        "current_only": 4.05045747756958,
        "current_only_val_best": 0.8161356449127197,
        "full_context_reference": 1.6090151071548462,
        "full_context_reference_val_best": 0.7929927706718445,
        "proxy_importance_topk": 1.3323458433151245,
        "proxy_importance_topk_val_best": 0.7874364852905273,
        "split_seed": 999
      }
    ],
    "proxy_gain_over_current": 0.5036110240228034,
    "proxy_topk_val": 1.4807484149932861,
    "target_variant": "future_delta_last_minus_current"
  }
]
```
