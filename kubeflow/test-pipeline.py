import kfp.dsl as dsl
from kfp.dsl import PipelineVolume
from kubernetes.client.models import V1VolumeMount, V1Volume, V1PersistentVolumeClaimVolumeSource

def echo_op(message):
    return dsl.ContainerOp(
        name='echo',
        image='alpine:3.12',
        command=['echo', message]
    )

@dsl.pipeline(
    name='Basic Pipeline',
    description='A basic Kubeflow Pipeline'
)
def basic_pipeline():
    step1 = echo_op('Hello, World!')
    step2 = echo_op('This is the second step')

if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(basic_pipeline, 'basic_pipeline.yaml')