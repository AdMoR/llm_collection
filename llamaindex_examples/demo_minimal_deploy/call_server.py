import sys
import llama_deploy


client = llama_deploy.Client()
session = client.sync.core.sessions.create()
task_id = session.run_nowait("echo_workflow", message=sys.argv[1])
r = session.get_task_result_stream(task_id)
print(r)
