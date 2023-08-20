    # INITIALIZING DEPLOYMENT:
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

# DEFINING DEPLOYMENT SPEC:
DeploymentSpec(
    flow='../lighting_9.py',
    name="lighting9",
    schedule=IntervalSchedule(interval=timedelta(minutes=5)),
    flow_runner=SubprocessFlowRunner(),
    parameters={
        "Appliance": "lighting", 
        "run_id": "35330982e1604d09b408b8c2cd15c9c4",
    },
    tags=["ml"]
)