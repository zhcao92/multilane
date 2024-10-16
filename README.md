# Planning model training for multilane scenario

This project simulates the multilane driving scenarios to train a planning model

## Files

- `multilane_data_generator.py`: Generate the data in multilane scenarios
- `model_training_transformer.py`: Model training
- `model_evaluator_transformer.py`: Model evaluation in multilan scenarios

## Run evaluation

Evaluate imitation model without front car
```sh
python model_evaluator_transformer.py --model_path='model/model_imitation.pth' --num_agents=0
```

Evaluate imitation model with front car
```sh
python model_evaluator_transformer.py --model_path='model/model_imitation.pth' --num_agents=1
```