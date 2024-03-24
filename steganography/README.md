# Steganography

This folder contains methods to hide a message in an image. To encode a message use the following command:

```
python3 encode.py -i path_to_the_image.png -m path_to_the_message.txt -b bit_plans_to_use
```

To decode the message, use the following command:

```
python3 decode.py -i path_to_the_image.png -b bit_plans_to_use
```