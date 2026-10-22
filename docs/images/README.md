# Documentation images

- `fib_describe.jpg`: the console plan of `fib8.describe()` (the README's
  first example).
- `live_board.gif`: the Live Board during the CIFAR-10 example. To make it
  again, run

  ```bash
  uv run --extra vision python src/examples/ml/cv/cifar/ico_cifar_complete_flow_mp.py --epochs 2 --limit 8000 --profile
  ```

  and, while the first epoch trains, capture the board URL it prints every
  second or two for ~30 s, e.g. with headless Chrome:

  ```bash
  chrome --headless=new --hide-scrollbars --window-size=1500,1050 \
    --virtual-time-budget=1500 --screenshot=f01.png "<url>"
  ```

  Paint the table's Signature column over with the card's background (it is
  noise in a picture of the profile), crop the frames to the run's progress
  and training fold, and join them (e.g. Pillow, 500 ms each, 64 colors).
