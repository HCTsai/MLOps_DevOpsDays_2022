'''
Created on 2022年6月20日

@author: Hsiao-Chien Tsai(蔡効謙)
'''
from minio import Minio
from minio.error import S3Error


def main():
    # Create a client with the MinIO server playground, its access key
    # and secret key.
    client = Minio(
        "10.56.211.125:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure = False
    )

    # Make 'asiatrip' bucket if not exist.
    found = client.bucket_exists("test")
    if not found:
        client.make_bucket("test")
    else:
        print("Bucket 'test' already exists")

    # Upload '/home/user/Photos/asiaphotos.zip' as object name
    # 'asiaphotos-2015.zip' to bucket 'asiatrip'.
    client.fput_object(
        "test", "test.pptx", "D:\\Downloads\\群創_中原資訊工業講座_蔡効謙.pptx",
    )
    print(
        "'D:\\Downloads\\群創_中原資訊工業講座_蔡効謙.pptx' is successfully uploaded as "
        "object 'test.pptx' to bucket 'test'."
    )


if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print("error occurred.", exc)