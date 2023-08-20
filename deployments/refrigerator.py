    # INITIALIZING DEPLOYMENT:
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

# DEFINING DEPLOYMENT SPEC:
DeploymentSpec(
    flow='../refregirator5.py',
    name="Refregirator_5",
    schedule=IntervalSchedule(interval=timedelta(minutes=5)),
    flow_runner=SubprocessFlowRunner(),
    parameters={
        "Appliance": "Refrigerator", 
        "run_id": "086e12807dd34c9a9fc77c005315d129",
    },
    tags=["ml"]
)