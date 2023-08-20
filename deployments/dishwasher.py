    # INITIALIZING DEPLOYMENT:
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

# DEFINING DEPLOYMENT SPEC:
DeploymentSpec(
    flow='../dishwasher6.py',
    name="Dishwasher6",
    schedule=IntervalSchedule(interval=timedelta(minutes=5)),
    flow_runner=SubprocessFlowRunner(),
    parameters={
        "Appliance": "Dishwasher6", 
        "run_id": "334dc4608f504985b5ed5774e71105e7",
    },
    tags=["ml"]
)
