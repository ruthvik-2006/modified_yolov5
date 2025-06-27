<html>
<p>
  Temp_fixed.cu -> modified_yolov5.cu
</p>
  <p>
  remaining -> individual layers. 
</p> 
<p>
  kinda dead lock thing!!
  if we write __device__ for the individual layers calling function -> it cannot be used because of extern since host cannot call directly to gpu...
  if we write global -> error ::  a __global__ function call must be configured -> from device to global call 
</p>
<p>
  The errors has been solved for the backbone architecture
</p>
<p>
no individual layer files used ... similar code of it with some manipulations has been made inorder to remove nan error for the output of the layer every individual layer has been integrated in the same file..    
</p>
<p>
  can transfer those layers into it respective files..
</p>
</html>
