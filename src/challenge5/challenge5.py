
# coding: utf-8

# In[86]:


import sys
get_ipython().system('sudo {sys.executable} -m pip install h5py --upgrade ')


# In[15]:


from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.webservice import Webservice
from azureml.core.image import ContainerImage
from azureml.core import Workspace
ws = Workspace.create(name='workspace-challenge5',
                      subscription_id='027f426f-daa5-4c21-bb89-1fdead1fe8cb',    
                      resource_group='myresourcegroup',
                      create_resource_group=True,
                      location='westus2' # or other supported Azure region  
                     )
ws.write_config()


# In[19]:


# register model 
from azureml.core import Run
model = Model.register(ws, model_name='objectmodelchal5', model_path='objectmodelchal5.h5')
print(model.name, model.id, model.version, sep = '\t')


# ## YAML EXPORT

# In[90]:


from azureml.core.conda_dependencies import CondaDependencies 

myenv = CondaDependencies()
myenv.add_conda_package("keras")
myenv.add_conda_package("pillow")
myenv.add_conda_package("numpy")
myenv.add_conda_package("requests")
myenv.add_conda_package("h5py")

with open("myenv.yml","w") as f:
    f.write(myenv.serialize_to_string())


# In[91]:


model=Model(ws, 'objectmodelchal5')
model.download(target_dir = '.')


# In[92]:


# configure the image
image_config = ContainerImage.image_configuration(execution_script="score.py", 
                                                  runtime="python", 
                                                  conda_file="myenv.yml")
image2 = ContainerImage.create(workspace=ws,
                                       name='adventureworkimage',
                                       models=[model],
                                       image_config=image_config)


# In[93]:


image2.wait_for_creation(show_output=True)


# In[94]:


image2.creation_state


# In[71]:


#image2.update_creation_state()


# In[24]:


# service = Webservice.deploy_from_model(workspace=ws,
#                                        name='adventureworkimage',
#                                        deployment_config=aciconfig,
#                                        models=[model],
#                                        image_config=image_config)
# service.wait_for_deployment(show_output=True)


# In[80]:


from azureml.core.webservice import AciWebservice

aciconfig = AciWebservice.deploy_configuration(cpu_cores = 2, 
                                               memory_gb = 2, 
                                               tags = {"data": "mnist", "type": "classification"}, 
                                               description = 'Image recognition')


# In[96]:


service = Webservice.deploy_from_image(workspace=ws,
                                       name='challenge5service3',
                                       deployment_config=aciconfig,
                                       image=image2)
service.wait_for_deployment(show_output=True)
print(service.state)


# In[97]:


print(service.get_logs())


# In[10]:


import sys 
get_ipython().system('{sys.executable} -m pip install azureml-core')


# In[12]:


get_ipython().system('pip upgrade adal')


# In[98]:




