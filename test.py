import os

current_app_path = os.path.dirname(os.path.abspath(__file__))
print(current_app_path)
local_temp_path = os.path.join(current_app_path, 'temp')
print(f'Temp Dir: {local_temp_path}')
