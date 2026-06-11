import subprocess
import tempfile
import os
import sys
import signal
import time
from typing import List, Dict, Tuple

class CodeExecutor:
    """安全执行代码的类"""
    
    def __init__(self, timeout=10, memory_limit=256):
        self.timeout = timeout  # 秒
        self.memory_limit = memory_limit  # MB
        
    def execute_python_code(self, code: str, test_cases: List[Dict]) -> Dict:
        """
        执行Python代码并测试
        
        返回:
        {
            "passed": bool,
            "num_passed": int,
            "total_tests": int,
            "error": str or None,
            "results": List[Dict]  # 每个测试用例的结果
        }
        """
        results = []
        passed_count = 0
        
        for i, test_case in enumerate(test_cases):
            temp_file = None  # 1. 提前初始化temp_file为None
            try:
                # 创建临时文件
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    try:  # 2. 内层try防护文件写入异常
                        # 准备完整代码（包括测试）
                        full_code = code + "\n\n"
                        
                        # 为每个测试用例创建测试代码
                        test_code = self.create_test_code(test_case)
                        full_code += test_code
                        
                        f.write(full_code)
                        f.flush()
                        temp_file = f.name
                    except Exception as e:
                        raise e  # 抛给外层处理
                
                # 执行代码（带超时限制）
                result = self.run_with_timeout(temp_file)
                
                if result["success"]:
                    test_passed = "TEST_PASSED" in result["output"]
                    if test_passed:
                        passed_count += 1
                        results.append({
                            "test_case": i,
                            "status": "passed",
                            "output": result["output"]
                        })
                    else:
                        results.append({
                            "test_case": i,
                            "status": "failed",
                            "output": result["output"],
                            "expected": test_case
                        })
                else:
                    results.append({
                        "test_case": i,
                        "status": "error",
                        "error": result["error"]
                    })
                    
            except Exception as e:
                results.append({
                    "test_case": i,
                    "status": "error",
                    "error": str(e)
                })
            finally:
                # 3. 先判断temp_file是否为None，再检查文件是否存在
                if temp_file is not None and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except Exception as e:
                        # 记录删除失败，但不中断程序
                        results[-1]["cleanup_error"] = f"删除临时文件失败: {str(e)}"
        
        return {
            "passed": passed_count == len(test_cases),
            "num_passed": passed_count,
            "total_tests": len(test_cases),
            "results": results
        }

    def run_with_timeout(self, file_path: str) -> Dict:
        """带超时限制的执行"""
        try:
            # 使用子进程执行
            result = subprocess.run(
                [sys.executable, file_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, 'PYTHONPATH': ''}  # 隔离环境
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Timeout after {self.timeout} seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e)
            }
    
    def create_test_code(self, test_case: Dict) -> str:
        """创建测试代码"""
        # MBPP 测试用例通常是断言语句
        test_string = test_case
        
        # 提取测试逻辑
        if "assert" in test_string:
            # 直接使用断言语句
            return f"\ntry:\n    {test_string}\n    print('TEST_PASSED')\nexcept Exception as e:\n    print(f'TEST_FAILED: {{e}}')"
        else:
            # 简单测试
            return f"\nresult = None  # 这里应该计算实际结果\nprint('TEST_RESULT:', result)"


def main():
    # 示例代码和测试用例
    test_list = [
    "assert first_repeated_char(\"abcabc\") == \"a\"",
    "assert first_repeated_char(\"abc\") == \"None\"",
    "assert first_repeated_char(\"123123\") == \"1\""
    ]
    code = """def first_repeated_char(str1): \n
    \tfor index,c in enumerate(str1): \n
    \t\tif str1[:index+1].count(c) > 1: \n
    \t\t\treturn "a" \n
    \treturn "None" \n"""

    executor = CodeExecutor(timeout=5, memory_limit=128)
    result = executor.execute_python_code(code, test_list)
    print(result)

if __name__ == "__main__":
    main()