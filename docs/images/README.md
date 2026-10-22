# Documentation images

- `fib_describe.jpg`: the console plan of `fib8.describe()` (the README's
  first example).
- `live_board.png`: the Live Board during the CIFAR-10 example. To make it
  again, run

  ```bash
  PYTHONPATH=src python src/examples/ml/cv/cifar/ico_cifar_complete_flow_mp.py --epochs 3 --limit 6000 --profile
  ```

  and, while the first epoch trains, capture the board URL it prints, e.g.
  with headless Chrome:

  ```bash
  chrome --headless=new --hide-scrollbars --force-device-scale-factor=2 \
    --window-size=1500,2200 --virtual-time-budget=8000 --screenshot=board.png "<url>"
  ```

  then crop it to the run's card.
