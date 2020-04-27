

# Play arround project for Arduino Nano 33 BLE SENSE 

_Just a few files to Captuer various motions from the IMU unit on the Arduino and process them using Tensorflow, in order to put a model back on the Arduino that will classify the motion._ 


## Content 

`/sketches/` - runs on the Arduino 

`/tensorflow/` - Python scripts to build and test the model 

`/data/` - a couple of small datasets recorded by myself 

`/model` - tf lite model that will be used on the Arduino. 

## Credit 

This is really just a bit further development on the nice tutorial here: 

 https://blog.arduino.cc/2019/10/15/get-started-with-machine-learning-on-arduino/ 

Origin of the code can be found here: https://github.com/arduino/ArduinoTensorFlowLiteTutorials 
