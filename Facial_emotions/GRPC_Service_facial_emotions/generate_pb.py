from grpc.tools import protoc

protoc.main(
    (
      '',
      '--proto_path=./protos/',
      '--python_out=.',
      '--grpc_python_out=.',
      './protos/facial_emotions.proto'
    )
)
#python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. calculator_sqr.proto