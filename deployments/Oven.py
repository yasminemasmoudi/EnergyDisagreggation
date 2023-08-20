    # INITIALIZING DEPLOYMENT:
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

# DEFINING DEPLOYMENT SPEC:
DeploymentSpec(
    flow='../oven3.py',
    name="Oven3",
    schedule=IntervalSchedule(interval=timedelta(minutes=5)),
    flow_runner=SubprocessFlowRunner(),
    parameters={
        "Appliance": "Oven", 
        "run_id": "55691a7d9b554cdca64309a8d2d29cce",
    },
    tags=["ml"]
)

