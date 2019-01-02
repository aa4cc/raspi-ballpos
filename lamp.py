import logging
import time
import RPi.GPIO as GPIO

logger = logging.getLogger(__name__)

pin=None
delay=0
logic=True

def init():
    if not pin:
        logger.info("No lamp pin selected, no lamp control")
        return

    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        print("Starting lamp....     ", end="", flush=True)
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, True);
        if delay:
            # Let the light warm up
            time.sleep(delay)
        print("OK")
    except RuntimeError:
        logger.error("Error importing RPi.GPIO!  This is probably because you need superuser privileges. You can achieve this by using 'sudo' to run your script")
    except ImportError:
        logger.error("Libratry RPi.GPIO not found, light controll not possible! You can install it using 'sudo pip3 install rpi.gpio' to install library")

def deinit():
    if not pin:
        return
    GPIO.output(pin, False);
    GPIO.cleanup()

def on():
    if not pin:
        raise RuntimeError("No lamp pin selected, no lamp control")
    GPIO.output(pin, logic)

def off():
    if not pin:
        raise RuntimeError("No lamp pin selected, no lamp control")
    GPIO.output(pin, logic)

