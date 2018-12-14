

```python
import sys
!sudo {sys.executable} -m pip install h5py --upgrade 
```

    [33mThe directory '/data/home/team15/.cache/pip/http' or its parent directory is not owned by the current user and the cache has been disabled. Please check the permissions and owner of that directory. If executing pip with sudo, you may want sudo's -H flag.[0m
    [33mThe directory '/home/team15/.cache/pip' or its parent directory is not owned by the current user and caching wheels has been disabled. check the permissions and owner of that directory. If executing pip with sudo, you may want sudo's -H flag.[0m
    Collecting h5py
    [?25l  Downloading https://files.pythonhosted.org/packages/8e/cb/726134109e7bd71d98d1fcc717ffe051767aac42ede0e7326fd1787e5d64/h5py-2.8.0-cp36-cp36m-manylinux1_x86_64.whl (2.8MB)
    [K    100% |████████████████████████████████| 2.8MB 20.3MB/s ta 0:00:01
    [?25hRequirement not upgraded as not directly required: six in /data/anaconda/envs/py36/lib/python3.6/site-packages (from h5py) (1.11.0)
    Requirement not upgraded as not directly required: numpy>=1.7 in /data/anaconda/envs/py36/lib/python3.6/site-packages (from h5py) (1.14.5)
    [31mtwisted 18.7.0 requires PyHamcrest>=1.9.0, which is not installed.[0m
    [31mazureml-contrib-brainwave 0.1.56 requires tensorflow>=1.6, which is not installed.[0m
    [31mazureml-train-widgets 0.1.56 has requirement azureml-core==0.1.56, but you'll have azureml-core 0.1.59 which is incompatible.[0m
    [31mazureml-train-core 0.1.56 has requirement azureml-core==0.1.56, but you'll have azureml-core 0.1.59 which is incompatible.[0m
    [31mazureml-sdk 0.1.56 has requirement azureml-core==0.1.56, but you'll have azureml-core 0.1.59 which is incompatible.[0m
    [31mazureml-pipeline-core 0.1.56 has requirement azureml-core==0.1.56, but you'll have azureml-core 0.1.59 which is incompatible.[0m
    [31mazureml-contrib-tensorboard 0.1.56 has requirement azureml-core==0.1.56, but you'll have azureml-core 0.1.59 which is incompatible.[0m
    [31mazureml-contrib-server 0.1.56 has requirement azureml-core==0.1.56, but you'll have azureml-core 0.1.59 which is incompatible.[0m
    [31mazureml-contrib-run 0.1.56 has requirement azureml-core==0.1.56, but you'll have azureml-core 0.1.59 which is incompatible.[0m
    [31mazureml-contrib-brainwave 0.1.56 has requirement azureml-core==0.1.56, but you'll have azureml-core 0.1.59 which is incompatible.[0m
    Installing collected packages: h5py
    Successfully installed h5py-2.8.0
    [33mYou are using pip version 10.0.1, however version 18.1 is available.
    You should consider upgrading via the 'pip install --upgrade pip' command.[0m



```python
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
```

    Wrote the config file config.json to: /data/home/team15/notebooks/challenge5/aml_config/config.json



```python
# register model 
from azureml.core import Run
model = Model.register(ws, model_name='objectmodelchal5', model_path='objectmodelchal5.h5')
print(model.name, model.id, model.version, sep = '\t')
```

    Registering model objectmodelchal5
    objectmodelchal5	objectmodelchal5:2	2


## YAML EXPORT


```python
from azureml.core.conda_dependencies import CondaDependencies 

myenv = CondaDependencies()
myenv.add_conda_package("keras")
myenv.add_conda_package("pillow")
myenv.add_conda_package("numpy")
myenv.add_conda_package("requests")
myenv.add_conda_package("h5py")

with open("myenv.yml","w") as f:
    f.write(myenv.serialize_to_string())
```


```python
model=Model(ws, 'objectmodelchal5')
model.download(target_dir = '.')
```




    './objectmodelchal5.h5'




```python
# configure the image
image_config = ContainerImage.image_configuration(execution_script="score.py", 
                                                  runtime="python", 
                                                  conda_file="myenv.yml")
image2 = ContainerImage.create(workspace=ws,
                                       name='adventureworkimage',
                                       models=[model],
                                       image_config=image_config)

```

    Creating image



```python
image2.wait_for_creation(show_output=True)
```

    Running......................................................
    SucceededImage creation operation finished for image adventureworkimage:7, operation "Succeeded"



```python
image2.creation_state
```




    'Succeeded'




```python
#image2.update_creation_state()
```


```python
# service = Webservice.deploy_from_model(workspace=ws,
#                                        name='adventureworkimage',
#                                        deployment_config=aciconfig,
#                                        models=[model],
#                                        image_config=image_config)
# service.wait_for_deployment(show_output=True)
```

    Creating image
    Image creation operation finished for image adventureworkimage:1, operation "Succeeded"
    Creating service



    ---------------------------------------------------------------------------

    HTTPError                                 Traceback (most recent call last)

    /anaconda/envs/py36/lib/python3.6/site-packages/azureml/core/webservice/webservice.py in _deploy_webservice(workspace, name, webservice_payload, webservice_class)
        335             resp = requests.post(mms_endpoint, params=params, headers=headers, json=webservice_payload)
    --> 336             resp.raise_for_status()
        337         except requests.exceptions.HTTPError:


    /anaconda/envs/py36/lib/python3.6/site-packages/requests/models.py in raise_for_status(self)
        938         if http_error_msg:
    --> 939             raise HTTPError(http_error_msg, response=self)
        940 


    HTTPError: 409 Client Error: Conflict for url: https://westus2.modelmanagement.azureml.net/api/subscriptions/027f426f-daa5-4c21-bb89-1fdead1fe8cb/resourceGroups/myresourcegroup/providers/Microsoft.MachineLearningServices/workspaces/workspace-challenge5/services?api-version=2018-03-01-preview

    
    During handling of the above exception, another exception occurred:


    WebserviceException                       Traceback (most recent call last)

    <ipython-input-24-3d7fda4da525> in <module>()
          3                                        deployment_config=aciconfig,
          4                                        models=[model],
    ----> 5                                        image_config=image_config)
          6 service.wait_for_deployment(show_output=True)


    /anaconda/envs/py36/lib/python3.6/site-packages/azureml/core/webservice/webservice.py in deploy_from_model(workspace, name, models, image_config, deployment_config, deployment_target)
        252         if image.creation_state != 'Succeeded':
        253             raise WebserviceException('Error occurred creating image {} for service.'.format(image.id))
    --> 254         return Webservice.deploy_from_image(workspace, name, image, deployment_config, deployment_target)
        255 
        256     @staticmethod


    /anaconda/envs/py36/lib/python3.6/site-packages/azureml/core/webservice/webservice.py in deploy_from_image(workspace, name, image, deployment_config, deployment_target)
        280                     if child._webservice_type == 'ACI':
        281                         return child._deploy(workspace, name, image, deployment_config)
    --> 282             return deployment_config._webservice_type._deploy(workspace, name, image, deployment_config)
        283 
        284         else:


    /anaconda/envs/py36/lib/python3.6/site-packages/azureml/core/webservice/aci.py in _deploy(workspace, name, image, deployment_config)
        142         deployment_config.validate_image(image)
        143         create_payload = AciWebservice._build_create_payload(name, image, deployment_config)
    --> 144         return Webservice._deploy_webservice(workspace, name, create_payload, AciWebservice)
        145 
        146     @staticmethod


    /anaconda/envs/py36/lib/python3.6/site-packages/azureml/core/webservice/webservice.py in _deploy_webservice(workspace, name, webservice_payload, webservice_class)
        339                                       'Response Code: {}\n'
        340                                       'Headers: {}\n'
    --> 341                                       'Content: {}'.format(resp.status_code, resp.headers, resp.content))
        342         if resp.status_code != 202:
        343             raise WebserviceException('Error occurred creating service:\n'


    WebserviceException: Received bad response from Model Management Service:
    Response Code: 409
    Headers: {'Date': 'Thu, 13 Dec 2018 21:10:23 GMT', 'Content-Type': 'application/json', 'Transfer-Encoding': 'chunked', 'Connection': 'keep-alive', 'api-supported-versions': '2018-03-01-preview, 2018-11-19', 'x-ms-client-request-id': '525eeec7e99d495f9b8b444d8a481132', 'x-ms-client-session-id': '', 'Strict-Transport-Security': 'max-age=15724800; includeSubDomains; preload'}
    Content: b'{"code":"Conflict","statusCode":409,"message":"Conflict","details":[{"code":"ServiceUnavailable","message":"The requested resource is not available in the location \'westus\' at this moment. Please retry with a different resource request or in another location. Resource requested: \'7\' CPU \'32.5\' GB memory \'Linux\' OS"}]}'



```python
from azureml.core.webservice import AciWebservice

aciconfig = AciWebservice.deploy_configuration(cpu_cores = 2, 
                                               memory_gb = 2, 
                                               tags = {"data": "mnist", "type": "classification"}, 
                                               description = 'Image recognition')
```


```python
service = Webservice.deploy_from_image(workspace=ws,
                                       name='challenge5service3',
                                       deployment_config=aciconfig,
                                       image=image2)
service.wait_for_deployment(show_output=True)
print(service.state)
```

    Creating service
    Running...............................
    FailedACI service creation operation finished, operation "Failed"
    Service creation failed, unexpected error response:
    {'code': 'AciDeploymentFailed', 'message': 'Aci Deployment failed', 'details': [{'code': 'CrashLoopBackOff', 'message': "Your container application crashed. This may be caused by errors in your scoring file's init() function.\nPlease check the logs for your container instance challenge5service3.\nYou can also try to run image workspacacrfkawxinj.azurecr.io/adventureworkimage:7 locally. Please refer to http://aka.ms/debugimage for more information."}]}
    Failed



```python
print(service.get_logs())
```

    2018-12-13T22:56:25,399160286+00:00 - iot-server/run 
    ok: run: gunicorn: (pid 14) 0s
    ok: run: nginx: (pid 11) 0s
    ok: run: rsyslog: (pid 12) 0s
    2018-12-13T22:56:25,400936482+00:00 - rsyslog/run 
    2018-12-13T22:56:25,398921487+00:00 - nginx/run 
    2018-12-13T22:56:25,401976779+00:00 - gunicorn/run 
    ok: run: rsyslog: (pid 12) 0s
    ok: run: rsyslog: (pid 12) 0s
    EdgeHubConnectionString and IOTEDGE_IOTHUBHOSTNAME are not set. Exiting...
    2018-12-13T22:56:25,473102901+00:00 - iot-server/finish 1 0
    2018-12-13T22:56:25,474107899+00:00 - Exit code 1 is normal. Not restarting iot-server.
    {"timestamp": "2018-12-13T22:56:25.693640Z", "message": "Starting gunicorn 19.6.0", "host": "wk-caas-0ec3f925831b490b9f00c6a3b9fa213e-a7a56cd8f33f11da05ac02", "path": "/opt/miniconda/lib/python3.6/site-packages/gunicorn/glogging.py", "tags": "%(module)s, %(asctime)s, %(levelname)s, %(message)s", "level": "INFO", "logger": "gunicorn.error", "msg": "Starting gunicorn %s", "stack_info": null}
    {"timestamp": "2018-12-13T22:56:25.694385Z", "message": "Listening at: http://127.0.0.1:9090 (14)", "host": "wk-caas-0ec3f925831b490b9f00c6a3b9fa213e-a7a56cd8f33f11da05ac02", "path": "/opt/miniconda/lib/python3.6/site-packages/gunicorn/glogging.py", "tags": "%(module)s, %(asctime)s, %(levelname)s, %(message)s", "level": "INFO", "logger": "gunicorn.error", "msg": "Listening at: %s (%s)", "stack_info": null}
    {"timestamp": "2018-12-13T22:56:25.694484Z", "message": "Using worker: sync", "host": "wk-caas-0ec3f925831b490b9f00c6a3b9fa213e-a7a56cd8f33f11da05ac02", "path": "/opt/miniconda/lib/python3.6/site-packages/gunicorn/glogging.py", "tags": "%(module)s, %(asctime)s, %(levelname)s, %(message)s", "level": "INFO", "logger": "gunicorn.error", "msg": "Using worker: %s", "stack_info": null}
    {"timestamp": "2018-12-13T22:56:25.695061Z", "message": "worker timeout is set to 300", "host": "wk-caas-0ec3f925831b490b9f00c6a3b9fa213e-a7a56cd8f33f11da05ac02", "path": "/opt/miniconda/lib/python3.6/site-packages/gunicorn/glogging.py", "tags": "%(module)s, %(asctime)s, %(levelname)s, %(message)s", "level": "INFO", "logger": "gunicorn.error", "stack_info": null}
    {"timestamp": "2018-12-13T22:56:25.696014Z", "message": "Booting worker with pid: 40", "host": "wk-caas-0ec3f925831b490b9f00c6a3b9fa213e-a7a56cd8f33f11da05ac02", "path": "/opt/miniconda/lib/python3.6/site-packages/gunicorn/glogging.py", "tags": "%(module)s, %(asctime)s, %(levelname)s, %(message)s", "level": "INFO", "logger": "gunicorn.error", "msg": "Booting worker with pid: %s", "stack_info": null}
    Initializing logger
    {"timestamp": "2018-12-13T22:56:29.748940Z", "message": "{\"requestId\": \"00000000-0000-0000-0000-000000000000\", \"message\": \"Starting up app insights client\", \"apiName\": \"\"}", "host": "wk-caas-0ec3f925831b490b9f00c6a3b9fa213e-a7a56cd8f33f11da05ac02", "path": "/var/azureml-app/aml_logger.py", "tags": "%(module)s, %(asctime)s, %(levelname)s, %(message)s", "level": "INFO", "logger": "root", "stack_info": null}
    {"timestamp": "2018-12-13T22:56:29.750106Z", "message": "{\"requestId\": \"00000000-0000-0000-0000-000000000000\", \"message\": \"Starting up request id generator\", \"apiName\": \"\"}", "host": "wk-caas-0ec3f925831b490b9f00c6a3b9fa213e-a7a56cd8f33f11da05ac02", "path": "/var/azureml-app/aml_logger.py", "tags": "%(module)s, %(asctime)s, %(levelname)s, %(message)s", "level": "INFO", "logger": "root", "stack_info": null}
    {"timestamp": "2018-12-13T22:56:29.750211Z", "message": "{\"requestId\": \"00000000-0000-0000-0000-000000000000\", \"message\": \"Starting up app insight hooks\", \"apiName\": \"\"}", "host": "wk-caas-0ec3f925831b490b9f00c6a3b9fa213e-a7a56cd8f33f11da05ac02", "path": "/var/azureml-app/aml_logger.py", "tags": "%(module)s, %(asctime)s, %(levelname)s, %(message)s", "level": "INFO", "logger": "root", "stack_info": null}
    {"timestamp": "2018-12-13T22:56:29.750321Z", "message": "{\"requestId\": \"00000000-0000-0000-0000-000000000000\", \"message\": \"Invoking user's init function\", \"apiName\": \"\"}", "host": "wk-caas-0ec3f925831b490b9f00c6a3b9fa213e-a7a56cd8f33f11da05ac02", "path": "/var/azureml-app/aml_logger.py", "tags": "%(module)s, %(asctime)s, %(levelname)s, %(message)s", "level": "INFO", "logger": "root", "stack_info": null}
    {"timestamp": "2018-12-13T22:56:29.751276Z", "message": "{\"requestId\": \"00000000-0000-0000-0000-000000000000\", \"message\": \"User's init function failed\", \"apiName\": \"\"}", "host": "wk-caas-0ec3f925831b490b9f00c6a3b9fa213e-a7a56cd8f33f11da05ac02", "path": "/var/azureml-app/aml_logger.py", "tags": "%(module)s, %(asctime)s, %(levelname)s, %(message)s", "level": "ERROR", "logger": "root", "stack_info": null}
    Using TensorFlow backend.
    {"timestamp": "2018-12-13T22:56:29.752141Z", "message": "{\"requestId\": \"00000000-0000-0000-0000-000000000000\", \"message\": \"Encountered Exception Traceback (most recent call last):\\n  File \\\"/var/azureml-app/aml_blueprint.py\\\", line 109, in register\\n    main.init()\\n  File \\\"/var/azureml-app/main.py\\\", line 79, in init\\n    driver_module.init()\\n  File \\\"score.py\\\", line 12, in init\\n    model = load_model('objectmodelchal5.h5')\\n  File \\\"/opt/miniconda/lib/python3.6/site-packages/keras/engine/saving.py\\\", line 417, in load_model\\n    f = h5dict(filepath, 'r')\\n  File \\\"/opt/miniconda/lib/python3.6/site-packages/keras/utils/io_utils.py\\\", line 186, in __init__\\n    self.data = h5py.File(path, mode=mode)\\n  File \\\"/opt/miniconda/lib/python3.6/site-packages/h5py/_hl/files.py\\\", line 312, in __init__\\n    fid = make_fid(name, mode, userblock_size, fapl, swmr=swmr)\\n  File \\\"/opt/miniconda/lib/python3.6/site-packages/h5py/_hl/files.py\\\", line 142, in make_fid\\n    fid = h5f.open(name, flags, fapl=fapl)\\n  File \\\"h5py/_objects.pyx\\\", line 54, in h5py._objects.with_phil.wrapper\\n  File \\\"h5py/_objects.pyx\\\", line 55, in h5py._objects.with_phil.wrapper\\n  File \\\"h5py/h5f.pyx\\\", line 78, in h5py.h5f.open\\nOSError: Unable to open file (unable to open file: name = 'objectmodelchal5.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)\\n\", \"apiName\": \"\"}", "host": "wk-caas-0ec3f925831b490b9f00c6a3b9fa213e-a7a56cd8f33f11da05ac02", "path": "/var/azureml-app/aml_logger.py", "tags": "%(module)s, %(asctime)s, %(levelname)s, %(message)s", "level": "ERROR", "logger": "root", "stack_info": null}
    {"timestamp": "2018-12-13T22:56:29.752386Z", "message": "Worker exiting (pid: 40)", "host": "wk-caas-0ec3f925831b490b9f00c6a3b9fa213e-a7a56cd8f33f11da05ac02", "path": "/opt/miniconda/lib/python3.6/site-packages/gunicorn/glogging.py", "tags": "%(module)s, %(asctime)s, %(levelname)s, %(message)s", "level": "INFO", "logger": "gunicorn.error", "msg": "Worker exiting (pid: %s)", "stack_info": null}
    {"timestamp": "2018-12-13T22:56:29.983932Z", "message": "Shutting down: Master", "host": "wk-caas-0ec3f925831b490b9f00c6a3b9fa213e-a7a56cd8f33f11da05ac02", "path": "/opt/miniconda/lib/python3.6/site-packages/gunicorn/glogging.py", "tags": "%(module)s, %(asctime)s, %(levelname)s, %(message)s", "level": "INFO", "logger": "gunicorn.error", "msg": "Shutting down: %s", "stack_info": null}
    {"timestamp": "2018-12-13T22:56:29.984298Z", "message": "Reason: Worker failed to boot.", "host": "wk-caas-0ec3f925831b490b9f00c6a3b9fa213e-a7a56cd8f33f11da05ac02", "path": "/opt/miniconda/lib/python3.6/site-packages/gunicorn/glogging.py", "tags": "%(module)s, %(asctime)s, %(levelname)s, %(message)s", "level": "INFO", "logger": "gunicorn.error", "msg": "Reason: %s", "stack_info": null}
    2018-12-13T22:56:30,008522386+00:00 - gunicorn/finish 3 0
    2018-12-13T22:56:30,009611284+00:00 - Exit code 3 is not normal. Killing image.
    



```python
import sys 
!{sys.executable} -m pip install azureml-core
```

    Collecting azureml-core
      Using cached https://files.pythonhosted.org/packages/f5/ce/3443d87ba4735de037003413063b7d5d77b00543b787c495b0ab682be254/azureml_core-1.0.2-py2.py3-none-any.whl
    Requirement already satisfied: SecretStorage<3.0.0 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (2.3.1)
    Collecting azure-cli-profile>=2.0.26 (from azureml-core)
      Using cached https://files.pythonhosted.org/packages/c2/d3/fdc722a1b61857250a76027d6d73a50182c6d85132ddd65600a8993574ce/azure_cli_profile-2.1.2-py2.py3-none-any.whl
    Requirement already satisfied: azure-storage-blob>=1.1.0 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (1.3.1)
    Requirement already satisfied: azure-mgmt-resource>=1.2.1 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (2.0.0)
    Requirement already satisfied: azure-storage-common>=1.1.0 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (1.3.0)
    Requirement already satisfied: ruamel.yaml<=0.15.51,>=0.15.35 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (0.15.51)
    Requirement already satisfied: azure-mgmt-containerregistry>=2.0.0 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (2.2.0)
    Collecting pathspec (from azureml-core)
    Requirement already satisfied: azure-graphrbac>=0.40.0 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (0.40.0)
    Requirement already satisfied: requests>=2.19.1 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (2.21.0)
    Requirement already satisfied: backports.tempfile in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (1.0)
    Requirement already satisfied: msrestazure>=0.4.33 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (0.5.0)
    Requirement already satisfied: PyJWT in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (1.6.4)
    Requirement already satisfied: azure-mgmt-authorization>=0.40.0 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (0.50.0)
    Requirement already satisfied: azure-mgmt-keyvault>=0.40.0 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (1.1.0)
    Requirement already satisfied: pytz in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (2018.4)
    Requirement already satisfied: urllib3<1.24,>=1.23 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (1.23)
    Collecting azure-cli-core>=2.0.38 (from azureml-core)
      Using cached https://files.pythonhosted.org/packages/a1/f0/507c83334d7ee7d588fdd1210b6f470446077c62f3e22fdb955f9d3396f8/azure_cli_core-2.0.52-py2.py3-none-any.whl
    Requirement already satisfied: python-dateutil>=2.7.3 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (2.7.3)
    Requirement already satisfied: docker in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (3.6.0)
    Requirement already satisfied: jsonpickle in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (1.0)
    Requirement already satisfied: ndg-httpsclient in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (0.5.1)
    Requirement already satisfied: contextlib2 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (0.5.5)
    Requirement already satisfied: six>=1.11.0 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (1.11.0)
    Requirement already satisfied: azure-common>=1.1.12 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (1.1.15)
    Requirement already satisfied: msrest>=0.5.1 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (0.5.5)
    Requirement already satisfied: azure-mgmt-storage>=1.5.0 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (2.0.0)
    Requirement already satisfied: cryptography!=1.9,!=2.0.*,!=2.1.*,!=2.2.* in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (2.3.1)
    Requirement already satisfied: azure-storage-nspkg>=3.0.0 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azureml-core) (3.0.0)
    Requirement already satisfied: azure-cli-command-modules-nspkg>=2.0.0 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azure-cli-profile>=2.0.26->azureml-core) (2.0.2)
    Requirement already satisfied: azure-mgmt-nspkg>=2.0.0 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azure-mgmt-resource>=1.2.1->azureml-core) (2.0.0)
    Requirement already satisfied: azure-nspkg>=2.0.0 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azure-graphrbac>=0.40.0->azureml-core) (2.0.0)
    Requirement already satisfied: idna<2.9,>=2.5 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from requests>=2.19.1->azureml-core) (2.6)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from requests>=2.19.1->azureml-core) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from requests>=2.19.1->azureml-core) (2018.8.24)
    Requirement already satisfied: backports.weakref in /data/anaconda/envs/py35/lib/python3.5/site-packages (from backports.tempfile->azureml-core) (1.0.post1)
    Requirement already satisfied: adal<2.0.0,>=0.6.0 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from msrestazure>=0.4.33->azureml-core) (1.1.0)
    Collecting pyyaml~=3.13 (from azure-cli-core>=2.0.38->azureml-core)
    Requirement already satisfied: argcomplete>=1.8.0 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azure-cli-core>=2.0.38->azureml-core) (1.9.4)
    Requirement already satisfied: pip in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azure-cli-core>=2.0.38->azureml-core) (18.0)
    Collecting humanfriendly>=4.7 (from azure-cli-core>=2.0.38->azureml-core)
      Using cached https://files.pythonhosted.org/packages/79/1e/13d96248e3fcaa7777b61fa889feab44865c85e524bbd667acfa0d8b66e3/humanfriendly-4.17-py2.py3-none-any.whl
    Collecting knack==0.5.1 (from azure-cli-core>=2.0.38->azureml-core)
      Using cached https://files.pythonhosted.org/packages/27/46/0a6d7471efcc519e392640f6933c0f644bbf602971e64797108292cb3623/knack-0.5.1-py2.py3-none-any.whl
    Requirement already satisfied: colorama>=0.3.9 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azure-cli-core>=2.0.38->azureml-core) (0.3.9)
    Requirement already satisfied: jmespath in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azure-cli-core>=2.0.38->azureml-core) (0.9.3)
    Requirement already satisfied: tabulate<=0.8.2,>=0.7.7 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azure-cli-core>=2.0.38->azureml-core) (0.8.2)
    Requirement already satisfied: antlr4-python3-runtime; python_version >= "3.0" in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azure-cli-core>=2.0.38->azureml-core) (4.7.1)
    Requirement already satisfied: wheel==0.30.0 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azure-cli-core>=2.0.38->azureml-core) (0.30.0)
    Requirement already satisfied: pyopenssl>=17.1.0 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azure-cli-core>=2.0.38->azureml-core) (18.0.0)
    Requirement already satisfied: azure-cli-telemetry in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azure-cli-core>=2.0.38->azureml-core) (1.0.0)
    Requirement already satisfied: pygments in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azure-cli-core>=2.0.38->azureml-core) (2.2.0)
    Requirement already satisfied: paramiko>=2.0.8 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azure-cli-core>=2.0.38->azureml-core) (2.4.2)
    Requirement already satisfied: azure-cli-nspkg>=2.0.0 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azure-cli-core>=2.0.38->azureml-core) (3.0.3)
    Requirement already satisfied: websocket-client>=0.32.0 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from docker->azureml-core) (0.54.0)
    Requirement already satisfied: docker-pycreds>=0.3.0 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from docker->azureml-core) (0.4.0)
    Requirement already satisfied: pyasn1>=0.1.1 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from ndg-httpsclient->azureml-core) (0.4.4)
    Requirement already satisfied: isodate>=0.6.0 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from msrest>=0.5.1->azureml-core) (0.6.0)
    Requirement already satisfied: requests-oauthlib>=0.5.0 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from msrest>=0.5.1->azureml-core) (1.0.0)
    Requirement already satisfied: asn1crypto>=0.21.0 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from cryptography!=1.9,!=2.0.*,!=2.1.*,!=2.2.*->azureml-core) (0.24.0)
    Requirement already satisfied: cffi!=1.11.3,>=1.7 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from cryptography!=1.9,!=2.0.*,!=2.1.*,!=2.2.*->azureml-core) (1.11.5)
    Requirement already satisfied: portalocker==1.2.1 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azure-cli-telemetry->azure-cli-core>=2.0.38->azureml-core) (1.2.1)
    Requirement already satisfied: applicationinsights>=0.11.1 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from azure-cli-telemetry->azure-cli-core>=2.0.38->azureml-core) (0.11.6)
    Requirement already satisfied: pynacl>=1.0.1 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from paramiko>=2.0.8->azure-cli-core>=2.0.38->azureml-core) (1.3.0)
    Requirement already satisfied: bcrypt>=3.1.3 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from paramiko>=2.0.8->azure-cli-core>=2.0.38->azureml-core) (3.1.5)
    Requirement already satisfied: oauthlib>=0.6.2 in /data/anaconda/envs/py35/lib/python3.5/site-packages (from requests-oauthlib>=0.5.0->msrest>=0.5.1->azureml-core) (2.1.0)
    Requirement already satisfied: pycparser in /data/anaconda/envs/py35/lib/python3.5/site-packages (from cffi!=1.11.3,>=1.7->cryptography!=1.9,!=2.0.*,!=2.1.*,!=2.2.*->azureml-core) (2.18)
    [31mazure-cli-core 2.0.52 has requirement adal>=1.2.0, but you'll have adal 1.1.0 which is incompatible.[0m
    Installing collected packages: pyyaml, humanfriendly, knack, azure-cli-core, azure-cli-profile, pathspec, azureml-core
      Found existing installation: PyYAML 3.12
    [31mCannot uninstall 'PyYAML'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.[0m
    [33mYou are using pip version 18.0, however version 18.1 is available.
    You should consider upgrading via the 'pip install --upgrade pip' command.[0m



```python
!pip upgrade adal
```

    /bin/sh: pip: command not found



```python

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-98-09a637d57aae> in <module>()
    ----> 1 h5py --version
    

    NameError: name 'h5py' is not defined

