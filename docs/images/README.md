# Documentation images

- `fib_describe.jpg`: the console plan of `fib8.describe()` (the README's
  first example).
- `live_board.png` and `live_board.gif`: the Live Board during the CIFAR-10
  example. To make them again, run

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

  For the GIF, capture a frame every second or two for ~30 s at
  `--window-size=1500,1050`, crop to the run's progress and training fold and
  join the frames (e.g. Pillow, 500 ms each, 64 colors).

  In both, the Signature column of the table is painted over with the card's
  background (it is noise in a picture of the profile).
