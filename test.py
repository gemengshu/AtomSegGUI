import matplotlib.pyplot as plt 
import serReader

ser = serReader.serReader('11.41.55 Scanning Acquire_1.ser')
imgplot = plt.imshow(ser['imageData'], cmap = 'gray')
plt.show()
print(ser['imageData'])