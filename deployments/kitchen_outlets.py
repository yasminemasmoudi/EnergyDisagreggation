    # INITIALIZING DEPLOYMENT:
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

# DEFINING DEPLOYMENT SPEC:
DeploymentSpec(
    flow='../kitchen_outlets7.py',
    name="kitchen_outlets_7",
    schedule=IntervalSchedule(interval=timedelta(minutes=5)),
    flow_runner=SubprocessFlowRunner(),
    parameters={
        "Appliance": "kitchen_outlets", 
        "run_id": "35330982e1604d09b408b8c2cd15c9c4",
    },
    tags=["ml"]
)