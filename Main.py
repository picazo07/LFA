import cv2
import matplotlib.pyplot as plt
import clasificador_covid_normal as cn
#%%
test = 4

#path = "test_images/COVID-"+str(test)+'.png' 
path = "test_images/Normal-"+str(test)+'.png'
image = cv2.imread(path,0)

clasificador = cn.clasificador_covid_normal()
prediction,probability,output_LFA = clasificador.clasificacion(image)

print("Radiograph classified as:",prediction)
print("COVID-19 probability:",probability)

#%% #######Mostrar imagen
image = cv2.equalizeHist(image) 
output_LFA = cv2.equalizeHist(output_LFA) 
plt.subplot(121)
plt.imshow(image,cmap = 'gray',vmin=0, vmax=255)
plt.title('Radiograph ')
plt.axis("off")
plt.subplot(122)    
plt.imshow(output_LFA,cmap = 'gray',vmin=0, vmax=255)
plt.title('ROI')
plt.axis("off")
plt.show()
