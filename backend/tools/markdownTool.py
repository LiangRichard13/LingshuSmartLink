import os 
from datetime import datetime
from tools.colorPrinter import ColorPrinter
import os
class markdownTool:
    def __init__(self)->None:
        current_working_dir = os.getcwd()  # 返回当前终端的工作目录
        self.file_path = current_working_dir+"/static/report"
    def save_markdown(self,markdown_content:str)->str:
        """
        将llm输出的内容转换为markdown格式并保存到指定路径的文件中。
        """

        #生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"markdown_{timestamp}.md"

        #构建完整的文件路径
        filepath = os.path.join(self.file_path, filename)
        
        # 写入文件
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            ColorPrinter.yellow("SystemMessage:")
            ColorPrinter.red(f"Markdown文件已成功保存到: {os.path.abspath(filepath)}")
            return 'static/report/'+filename
        
        except Exception as e:
            return f"保存报告文件时出错: {e}"