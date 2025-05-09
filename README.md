# 🧠 How-image-based-LLM-work?
In the previous article, we learned how text-based LLMs work.

👉 If you haven’t read it yet, check it out here: [https://github.com/trishnabhattarai/how-does-text-based-LLM-work] under Readme file.
To understand this article, Now we need to know:
- The LLMs which can understand text and image both are called Multimodal LLMs or Vision Transformers (ViTs).
- Vision Transformers (ViTs) are specially only used for image based VLMs.
- And the model which can only understand image are called VLM.

## 1. Patch Embedding:
   - When user upload image on any Visual Language Model (VLM) then algorithmns like Convolutional neutral network (CNN), Transformer or Multi- layer perceptrons (MLPs) mixer are used.
   - These algorithm divide that image into patches and pixels which is similar to tokens and sub tokens respectively.
   - Remember the length and width of square patches are fixed during training of the model. It doesn't depend upon the size of image.
   - Now, VLM identify the color and shape that patches contain.

![Screenshot 2025-05-08 222811](https://github.com/user-attachments/assets/c140f3e6-b898-4637-be4f-72816daeeb1d)

### Is pixels has similar use case as sub tokens has?
- Now if you remember, You need to have one question in your mind i.e. in LLM we learned the concept of subtoken is used when LLM need to understand the meaning of complex or unseen word
   - But in VLM, The concept of pixels are not used to understand the strcture or color of the complex image.

### What if VLM see new image that isn't avaliable in his dataset is there any backup plain like subtokens in LLM?
- 👉 No, if VLM see image of a new creature that isn't avaliable in his dataset then it will try to match the physical properties like Fur, eyes, tail, or legs to similar patterns it 
     has already seen before (like a dog, dragon, or wolf) then it will say
  >      "This looks like a hybrid between a wolf and a dragon."

  
## 2. Linear Projection Layer:
- Now, patches are send to linear projection layer which convert patches into vectors like in text based LLM, 1 word= 1 token= 1 vector so same as 1 patches= 1 vector. 
- Vector number is generated according to the avaliable information in patch.
- All the vector numbers are unique because every patch contain different color with different shape and size. 
- This vector number depends on brightness of the color, shape of the object and location of the object in image.
- The location of  object is determined by using mathematical concept like sin and cosine.
- Every position and color of object in a image is different so, the vector number are always unique.

![Linear Projection Layer_ - visual selection](https://github.com/user-attachments/assets/de77e3f8-78db-48c4-8899-1d3e6cc7ef3b)


### CLS Token:
In this step, When user upload his/ her image in VLM then CLS token is created where all the information of the image or patches has been stored in vector form like which patch contain 
which shapes or color etc.


## 3. Transformer (Neural Network):
- After that, Vectors are send to transformer where transfer tried to understand the relation between the patches with the help of self attention and feed- forward layer.
- Self Attention Layer helps VLM to compare the patches and identify which patch is the another part of patch.
- Feed- Forward Layer helps self attention layer to have deep research according to the information stored in CLS token.
- The patches which has less difference in vector number are considered as highly related to each other like this transformer can known the relation between patches.
![Transformer (Neural Network)_ - visual selection](https://github.com/user-attachments/assets/001e3381-744a-4734-9275-d65392c44e02)

## 4. Positional Encoding:
- Images are 2D model so position of each patch matter.
- If you are thinking this information is also stored in CLS token then you are thinking wrong.
- Positional Encoding will automatically identify the position of patch as it use mathmatical concept like sin and cosin with which it will automatically detect the position of patch.
- Remember, Vector number is generated according to the position of the patch so by the help of this vector number Positional Encoding automatically detect the position of patch.

![Positional Encoding_ - visual selection](https://github.com/user-attachments/assets/101c549a-9eec-4c27-9e1a-0fbdedb4edc9)


## 5. Binary Conversion:
- Now, The vector numbers are converted into binary number where numbers are represented in 0 and 1 form.
- Each o and 1 represent 1 bit.

![Binary Conversion_ - visual selection](https://github.com/user-attachments/assets/27b0a730-b912-4bd3-ac6c-d14a89981bcf)

### Do you know why we need more memory to train this type of AI model?
- Now your answer will be probabily: Because we train AI model on various data set so it take much memory.
- Ok this answer is very common have you tried to explore more deep?
- To understand that you need to know why we use GPU?
  - GPU is used to execute the program temperorly like when we write hello world program in any programming language and when you execute it.
  - It runs with the help of GPU and doesn't becomes overload on memory or on your system.
- Ok now let me explain you
  - We need to train data set so we need more memory.
  - I agree with that but in AI model user input data is compared with the pretrain data according to which unique vector number is generated.
  - Now, The vector number is not directly understand by machines so this vector number need to converted into binary number which take huge memory. (didn't understand?)

 ### (How Normal Computer or Machine Understand?)
- Ok let me clear you
  - In below, You can see a table contain Letter, ASCII Code and Binary field.
  - Every character has unique ASCII Code.
  - To understand the character, Computer replace the character with their ASCII value.
  - After that, It convert that ASCII code into binary number

```bash
Charcter ASCII Code    Binary            Character  ASCII Code     Binary
  a       097         01100001               A       065         01000001
  b       098         01100010               B       066         01000010
  c       099         01100011               C       067         01000011
  d       100         01100100               D       068         01000100
  e       101         01100101               E       069         01000101
  f       102         01100110               F       070         01000110
  g       103         01100111               G       071         01000111
  h       104         01101000               H       072         01001000
  i       105         01101001               I       073         01001001
  j       106         01101010               J       074         01001010
  k       107         01101011               K       075         01001011
  l       108         01101100               L       076         01001100
  m       109         01101101               M       077         01001101
  n       110         01101110               N       078         01001110
  o       111         01101111               O       079         01001111
  p       112         01110000               P       080         01010000
  q       113         01110001               Q       081         01010001
  r       114         01110010               R       082         01010010
  s       115         01110011               S       083         01010011
  t       116         01110100               T       084         01010100
  u       117         01110101               U       085         01010101
  v       118         01110110               V       086         01010110
  w       119         01110111               W       087         01010111
  x       120         01111000               X       088         01011000
  y       121         01111001               Y       089         01011001
  z       122         01111010               Z       090         01011010
```

- Example: If computer tries to understand "What is Botpress?" then it replace each character with its ASCII code and convert that code into binary number as you can see below.

```bash
W = 087     01010111              i = 105       01101001              B = 066     01000010              ? = 063    00111111
h = 104     01101000              s = 115       01110011              o = 111     01101111
a = 097     01100001                                                  t = 116     01110100
t = 116     01110100                                                  p = 112     01110000
                                                                      r = 114     01110010
                                                                      e = 101     01100101
                                                                      s = 115     01110011
                                                                      s = 115     01110011

                                               What                             is
                                                |                                |
Computer understand the question is:'01010111 01101000 01100001 01110100' '01101001 01110011'

                                  Botpress                                    ?
                                     |                                        |
'01000010 01101111 01110100 01110000 01110010 01100101 01110011 01110011‘ ‘00111111’
```
### (How AI model Understand?)

Remember:<br>
&nbsp;&nbsp;&nbsp;&nbsp; - AI model compares user input data with pretrain data according to which a unique vector number is generated.<br>
&nbsp;&nbsp;&nbsp;&nbsp; - The vector number is not directly understood by machines.<br>
&nbsp;&nbsp;&nbsp;&nbsp; - So, This vector number needs to be converted into a binary number which takes huge memory.

 Now, Lets see How does vector number in AI model looks like for same "What is Botpress?" sentence :
```bash
[
 0.45, -0.12, 0.33, 0.21, -0.67, 0.89, -0.45, 0.72, -0.34, 0.15, -0.78, 0.64, -0.91, 0.33, -0.26, 0.17, 0.53, -0.39, 0.12,
 0.55, 0.44, -0.13, 0.78, -0.29, 0.63, -0.15, 0.28, -0.49, 0.81, 0.02,0.41, -0.54, 0.11, 0.35, -0.83, 0.66, -0.34, 0.48,
 -0.22, 0.57, -0.76, 0.18, 0.95, -0.28, 0.43, -0.56, 0.71, -0.39, 0.27, 0.68, 0.55, -0.31, 0.89, -0.17, 0.32, -0.44, 0.23,
 -0.51, 0.72, -0.61, 0.64, -0.42, 0.12, 0.81, -0.19, 0.38, -0.75, 0.56, -0.84, 0.23,0.39, -0.45, 0.68, -0.17, 0.91, -0.31,
 0.22, 0.72, -0.12, 0.53, -0.62, 0.49, -0.14, 0.67, -0.29, 0.31, 0.79, -0.36, 0.58, -0.48,  0.85, -0.21, 0.74, -0.63, 0.11,
 0.93, -0.57, 0.27, -0.42, 0.76,0.14, -0.68, 0.53, -0.32, 0.81, -0.29, 0.48, -0.73, 0.65, -0.38, -0.15, 0.92, -0.31, 0.43,
 -0.64, 0.59, -0.52, 0.37, 0.86, -0.28, 0.74, -0.39, 0.58, -0.22, 0.63, -0.81, 0.49, -0.33, 0.62, -0.57, 0.75, -0.23, 0.92,
 -0.14, 0.56, -0.41, 0.89, -0.36, 0.77, -0.48,0.62, -0.25, 0.94, -0.57, 0.34, 0.72, -0.43, 0.86, -0.18, 0.51, -0.69, 0.42,
 -0.33, 0.87, -0.15, 0.58, -0.79, 0.36, -0.28, 0.93,0.45, -0.34, 0.76, -0.13, 0.51, -0.27, 0.62, -0.81, 0.47, -0.32, 0.94,
 -0.65, 0.29, -0.47, 0.71, -0.24, 0.84, -0.52, 0.39, 0.76, -0.58, 0.23, -0.91, 0.67, -0.39, 0.52, -0.48, 0.81, -0.29, 0.66,
 -0.44, 0.57, -0.71, 0.36, -0.84, 0.92, -0.25, 0.58, -0.61, 0.39,  -0.48, 0.87, -0.15, 0.62, -0.79, 0.33, -0.27, 0.92, -0.54,
 0.41, 0.82, -0.31, 0.73, -0.24, 0.56, -0.68, 0.43, -0.51, 0.77, -0.39, 0.93, -0.62, 0.49, -0.25, 0.88, -0.17, 0.53, -0.45,
 0.71, -0.28, -0.61, 0.36, -0.93, 0.52, -0.47, 0.75, -0.14, 0.68, -0.83, 0.25
]

Vector Number in Binary:

00111110111001100110011001100110
10111101111101011100001010001111
00111110101010001111010111000011
00111110010101110000101000111101
10111111001010111000010100011111
00111111011000111101011100001010
10111110111001100110011001100110
00111111001110000101000111101100
10111110101011100001010001111011
00111110000110011001100110011010
... (Just an example How does it look like)

```
- This is the vector and binary presentation of "What is Botpress?"
- AI system won't directly understand this number it need to convert into binary.
- Now imagine each 1 and 0 represnt 1 bit then how much GPU will be used to run yea, just to run this model?
  (Isn't it amazing.)
thats why tools like kaggle provide you virtual GPU memory to train your AI model.


### Don't you think patches effect the output of the VLM?
 - Yes, Patches effect the output of the VLM because more patches means more information for VLM to understand the image.<br>
 - Like more token means more information to understand the user requirement to generate text.<br>
 - It means 'High patches means high accuracy of image generation'.
