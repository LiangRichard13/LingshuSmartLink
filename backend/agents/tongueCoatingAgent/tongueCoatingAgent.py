from langchain.prompts import SystemMessagePromptTemplate
from dotenv import load_dotenv
import os
from agents.tongueCoatingAgent.tongueCoatingClassifier import TongueCoatingClassifier
from tools.colorPrinter import ColorPrinter
import base64
import openai


load_dotenv()   #加载环境变量

class TongueCoatingAgent:
    def __init__(self)->None:
        self.api_key=os.getenv("STEP_FUN_API_KEY") #从环境变量中获取API密钥
        self.model="step-1o-turbo-vision" #指定模型名称
        self.client=openai.Client(api_key=self.api_key,base_url="https://api.stepfun.com/v1") #调用stepfun的多模态大模型
        self.system_prompt_template="""
            你是一个舌苔诊断专家，你的任务是帮助用户诊断舌苔问题。\n
            你需要根据用户提供的舌苔病理图片、舌苔分类结果和置信度来进行综合诊断。\n
            请描述舌苔的特征（如舌质、舌苔颜色厚薄等）和可能的健康问题。\n
            不要给出任何医疗建议或治疗方案。\n
            """
        self.system_prompt=SystemMessagePromptTemplate.from_template(template=self.system_prompt_template)
    def llm_call(self,tongue_coating_result:str,confidence_score:str,image_path:str)->str:
        try:
            with open(image_path, "rb") as image_file:
                #获取图片类型
                image_type = image_path.split('.')[-1]
                if image_type not in ['jpg', 'jpeg', 'png']:
                    raise ValueError("Unsupported image format. Please use jpg, jpeg, or png.")
                #读取图片并转换为base64编码
                base64_bytes = base64.b64encode(image_file.read())
        
                completion = self.client.chat.completions.create(
                model="step-1v-8k",
                messages=[
                {
                    "role": "system",
                    "content": self.system_prompt_template,
                },
                # 在对话中传入图片，来实现基于图片的理解
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"以下是分类结果:{tongue_coating_result}\n和置信度:{confidence_score}%\n请根据这些信息进行诊断。\n",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_type};base64,{base64_bytes.decode('utf-8')}",
                            },
                        },
                    ],
                },
            ],
            )
        except FileNotFoundError: # 捕获文件未找到异常
            ColorPrinter.red("SystemMessage:")
            ColorPrinter.yellow(f"错误: 文件 {image_path} 没找到")
        except ValueError as ve:  # 捕获值错误
            ColorPrinter.red("SystemMessage:")
            ColorPrinter.yellow(f"错误: {ve}")
        except Exception as e: # 捕获其他异常
            ColorPrinter.red("SystemMessage:")
            ColorPrinter.yellow(f"错误: 未知错误: {e}")
        # 返回模型的输出
        return completion.choices[0].message.content
    def to_base64(self,image_path:str)->str:
        with open(image_path, "rb") as image_file:
            base64_bytes = base64.b64encode(image_file.read())
        return base64_bytes.decode("utf-8")

    def tongue_coating_diagnosis(self,image_path:str)->str:
        classifier=TongueCoatingClassifier() #创建舌苔分类器实例
        tongue_coating_result,confidence_score=classifier.predict_image(image_path) #对图像进行舌苔分类
        #调用LLM进行舌苔诊断
        response_content=self.llm_call(
            tongue_coating_result=tongue_coating_result,
            confidence_score=confidence_score,
            image_path=image_path) 
        return response_content
