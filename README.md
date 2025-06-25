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
</html>
