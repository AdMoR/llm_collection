## Minimal deploy example 

### Steps to deploy

Install the python env
```shell
python3 -m venv venv
source venv/bin/activate
pip install poetry
poetry install
```

Setup the workflow
- Create `workflow.py` with a WorkFlow 
- Define the configuration deployment file
  - It should link the right file to load th right object

```python
# `echo_workflow` will be imported by Llama Deploy
echo_workflow = EchoWorkflow()
```

```yaml
services:
  echo_workflow:
    name: Echo Workflow
    # We tell Llama Deploy where to look for our workflow
    source:
      # In this case, we instruct Llama Deploy to look in the local filesystem
      type: local
      # The path in the local filesystem where to look. This assumes there's an src folder in the
      # current working directory containing the file workflow.py we created previously
      name: ./
    # This assumes the file workflow.py contains a variable called `echo_workflow` containing our workflow instance
    path: workflow:echo_workflow
```

- Run the main server : `poetry run python -m llama_deploy.apiserver`
- Run the workflow, attached to this server
  - poetry run llamactl deploy deploy_config.yaml

### Checks in production

Is it alive 
- `poetry run llamactl status`

Send a request
- `poetry run python call_server.py Salut`


