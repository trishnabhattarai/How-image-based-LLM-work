# 🧠 How-image-based-LLM-work?
- In previous article, We learned how text based LLMs work.
- If you didn't read it. You can read it from the given link:[https://github.com/trishnabhattarai/how-does-text-based-LLM-work] under Readme file.
## To understand this article, Now we need to know:
- The LLMs which can understand text and image both are called Multimodal LLMs or Vision Transformers (ViTs) where Vision Transformers (ViTs) are specially only used for image based VLMs.
- And the model which can only understand image are called VLM.
### 1. Patch Embedding:
   - When user upload image on any Visual Language Model (VLM) then algorithmns like Convolutional neutral network (CNN), Transformer or Multi- layer perceptrons (MLPs) mixer divide that image
     patches and pixels which is similar to tokens and sub tokens respectively.
   - Remember the length and width of square patches are fixed during training of the model. It doesn't depend upon the size of image.
   - Now, VLM identify the color and shape that patches contain.

### Is pixels has similar use case as sub tokens has?
now if u remember, u need to have one question in your mind i.e. in LLM we learned the concept of subtoken is used when LLM need to understand the meaning of complex or unseen word
but in VLM the concept of pixels are not used to understand the strcture or color of the complex image.
### What if VLM see new image that isn't avaliable in his dataset is there any backup plain like subtokens in LLM?
No, if VLM see image of a new create that isn't avaliable in his dataset then it will try to match the physical properties like Fur, eyes, tail, or legs to similar patterns it has seen
before (like a dog, dragon, or wolf) then it will say "This looks like a hybrid between a wolf and a dragon."
2. linear projection layer:
now patch are send to linear projection layer which convert patches into vectors Like in text based LLM, 1 word= 1 token= 1 vector so same as 1 patches= 1 vector. Vector number is 
generated according to the avaliable information in patch as all the vector numbers are unique because every patch contain different color with different shape and size. this vector 
number depends on brightness of the color, shape of the object and location of the object using mathematical concept like sin and cosine so the vector number are always unique in a 
image.
#### CLS token is created
in this step when user upload his/ her image in VLM then CLS token is created where all the information of the image or patches has been stored in vector form like which patch contain 
which shapes or color etc.
3. Transformer (Neural Network):
after that vectors are send to transformer where transfer tried to understand the relation between the patches with the help of self attention and feed- forward layer where self attention
layer helps VLM to compare the patches and identify which patch is the another part of patch and feed- forward layer helps self attention layer to have deep research according to the 
information stored in CLS token. The patches which has less difference of vector number are considered as highly related to each other like this transformer can known the relation between
patches.
4. Positional encoding:
Images are 2D model so position of each patch matter. if u are thinking this information is also stored in CLS token then u are thinking wrong. Positional Encoding will automatically 
identify the position of patch as it use mathmatical concept like sin and cosin with which it will automatically detect the position of patch. Remember, vector number is generated 
according to the position of the patch so by the help of this vector number Positional encoding automatically detect the position of patch.

5. Binary Conversion:
   now the vector numbers are converted into binary number where numbers are represented in 0 and 1 form. every o and 1 represent 1 bit.

### Do you know why we need more memory to train this try of AI model?
now your answer will be probabily: because we train ai model on various data set so it take much memory.
ok this answer is very common have u tried to explore more deep?
to understand that u need to know why we use GPU? GPU is used to execute the program temperorly like when we write hello world program in any programming language when u execute it. it 
runs with the help of GPU and doesn't becomes overload on memory or on your system.
ok now let me explain u 
we need to train data set so we need more memory i agree with that but in AI model user input data is compared with the pretrin data according to which unique vector number is generated 
now the vector number is not directly understand by machines then this vector number need to converted into binary number which take huge memory (didn't understand?)
ok let me clear you
![image](https://github.com/user-attachments/assets/64d4d0c7-f79f-4eb9-8770-222101479f36)
![image](https://github.com/user-attachments/assets/c860acbd-6232-424e-851a-76c89cfabd0b)
![image](https://github.com/user-attachments/assets/f1dfd205-5036-4c49-adea-0e62793de13a)
![image](https://github.com/user-attachments/assets/181eb785-a46a-4d04-8378-6cbe24420ae0)
It means the Question is :'01010111 01101000 01100001 01110100' '01101001 01110011' '01000010 01101111 01110100 01110000 01110010 01100101 01110011 01110011‘ ‘00111111’

in above image u can see the binary conversion of "What is Botpress?" now each character has its own ASCII value like the ASCII code of A is 65 now convert 065 into binary i.e. 01000001
so in binary A means 01000001 but small a has another ASCII value ok u can see ASCII code for each character with its binary number in below image.
![Screenshot 2025-05-07 013000](https://github.com/user-attachments/assets/e42006d3-cc94-45b9-b105-ec0d992450a4)

 now lets see how does vector number in AI model looks like for same "What is Botpress?" sentence
[0.45, -0.12, 0.33, 0.21, -0.67, 0.89, -0.45, 0.72, -0.34, 0.15, -0.78, 0.64, -0.91, 0.33, -0.26, 0.17, 0.53, -0.39, 0.12, 0.55, 0.44, -0.13, 0.78, -0.29, 0.63, -0.15, 0.28, -0.49,
0.81, 0.02,0.41, -0.54, 0.11, 0.35, -0.83, 0.66, -0.34, 0.48, -0.22, 0.57, -0.76, 0.18, 0.95, -0.28, 0.43, -0.56, 0.71, -0.39, 0.27, 0.68, 0.55, -0.31, 0.89, -0.17, 0.32, -0.44, 0.23,
-0.51, 0.72, -0.61, 0.64, -0.42, 0.12, 0.81, -0.19, 0.38, -0.75, 0.56, -0.84, 0.23,0.39, -0.45, 0.68, -0.17, 0.91, -0.31, 0.22, 0.72, -0.12, 0.53, -0.62, 0.49, -0.14, 0.67, -0.29, 0.31,
0.79, -0.36, 0.58, -0.48,  0.85, -0.21, 0.74, -0.63, 0.11, 0.93, -0.57, 0.27, -0.42, 0.76,0.14, -0.68, 0.53, -0.32, 0.81, -0.29, 0.48, -0.73, 0.65, -0.38, -0.15, 0.92, -0.31, 0.43, -0.64,
0.59, -0.52, 0.37, 0.86, -0.28, 0.74, -0.39, 0.58, -0.22, 0.63, -0.81, 0.49, -0.33, 0.62, -0.57, 0.75, -0.23, 0.92, -0.14, 0.56, -0.41, 0.89, -0.36, 0.77, -0.48,0.62, -0.25, 0.94, -0.57,
0.34, 0.72, -0.43, 0.86, -0.18, 0.51, -0.69, 0.42, -0.33, 0.87, -0.15, 0.58, -0.79, 0.36, -0.28, 0.93,0.45, -0.34, 0.76, -0.13, 0.51, -0.27, 0.62, -0.81, 0.47, -0.32, 0.94, -0.65, 0.29, 
-0.47, 0.71, -0.24, 0.84, -0.52, 0.39, 0.76, -0.58, 0.23, -0.91, 0.67, -0.39, 0.52, -0.48, 0.81, -0.29, 0.66, -0.44, 0.57, -0.71, 0.36, -0.84, 0.92, -0.25, 0.58, -0.61, 0.39,  -0.48, 0.87,
-0.15, 0.62, -0.79, 0.33, -0.27, 0.92, -0.54, 0.41, 0.82, -0.31, 0.73, -0.24, 0.56, -0.68, 0.43, -0.51, 0.77, -0.39, 0.93, -0.62, 0.49, -0.25, 0.88, -0.17, 0.53, -0.45, 0.71, -0.28, -0.61,
0.36, -0.93, 0.52, -0.47, 0.75, -0.14, 0.68, -0.83, 0.25]
o my god! don't get nervious. this is the vector presentation of "What is Botpress?" AI system won't directly understand this number it need to convert into binary. Now imagine 1 or 0 
represnt 1 bit then how much GPU will be used to run just to run this model? Isn't it amazing.
thats why tools like kaggle provide u virtual GPU memory to train your AI model.
### don't u think patches effect the output of the VLM?
Yes, patches effect the output of the VLM because more patches means more information for VLM to understand the image like more token means more information to understand the user
requirement to geneerate text.
 ## high patches means high accuracy of image generation.
