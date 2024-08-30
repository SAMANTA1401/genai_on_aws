## this python file need to be input as lambda_function.py inside aws lambda and add layer of boto3 library as 
# zip file and connect with api gateway 

import boto3
import botocore.config
import json
from datetime import datetime

def blog_generate_using_bedrock(blogtopic:str)->str:
    prompt=f"""<s>[INST]Human: Write a 200 words blog on the topic {blogtopic}
    Assistant:[/INST]

    """
    body={
        "prompt":prompt,
        "max_gen_len":512,
        "temperature":0.5,
        "top_p":0.9
    }

    try:
        bedrock=boto3.client("bedrock-runtime",region_name='ap-south-1',config=botocore.config.Config(read_timeout=300,retries={
            'max_attempts':3
        }))
        print("boto3 accessed")
        response = bedrock.invoke_model(body=json.dumps(body),modelId="meta.llama3-8b-instruct-v1:0")
        print("response send")
        response_content=response.get('body').read()
        print("response read")
        response_data = json.loads(response_content)
        print(response_data)
        blog_details = response_data['generation']
        return blog_details
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while generating the blog."

def save_blog_details_s3(s3_key,s3_bucket,generate_blog):
    s3 = boto3.client('s3')

    try:
        s3.put_object(Bucket = s3_bucket, Key = s3_key, Body=generate_blog)
        print(f"Blog saved to S3 bucket: {s3_bucket}, Key: {s3_key}")

    except Exception as e:
        print(f"Error saving blog to S3: {e}")


def lambda_handler(event, context):
    # TODO implement
    # event = json.dumps(event['body'])
    event = json.loads(event['body'])
    
    print(event)
    # topic = json.loads(event)
    # print(topic)
    # blogtopic = topic['blog_topic']
    blogtopic = event['blog_topic']
    print(blogtopic)
    generate_blog = blog_generate_using_bedrock(blogtopic=blogtopic)
    if generate_blog:
        current_time = datetime.now().strftime('%H%M%S')
        s3_key = f"blog-output/{current_time}.txt"
        s3_bucket = 'shu-genai-blog'
        save_blog_details_s3(s3_key,s3_bucket,generate_blog)
    else:
        print("Error in generating the blog")

    return{
        "statusCode": 200,
        "body": json.dumps("Blog generated and saved to S3.")
    }

   
