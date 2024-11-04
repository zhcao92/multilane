# Planning model training for multilane scenario

This project simulates the multilane driving scenarios to train a planning model

## Files

- `multilane_data_generator.py`: Generate the data in multilane scenarios
- `model_training_transformer.py`: Model training
- `model_evaluator_transformer.py`: Model evaluation in multilan scenarios

## Run training
```sh
python model_training_L2L.py
```

## Run evaluation

Evaluate L2L model
```sh
python model_evaluator_L2L.py --model_path='best_model_L2L.pth' --ego_desired_speed=60 --surrounding_car=30 --num_agents=0 --only_longitudinal
```