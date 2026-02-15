import os
import argparse

def update_server_with_new_model(server_path, model_path, is_default=False):
    """
    Updates the server.py file to include the new model
    and allow switching between models.
    """
    if not os.path.exists(server_path):
        print(f"Error: Server file not found at {server_path}")
        return False
    
    # Read the current server file
    with open(server_path, 'r') as f:
        lines = f.readlines()
    
    # Find the MODEL_PATH line
    model_path_line_index = None
    for i, line in enumerate(lines):
        if "MODEL_PATH = " in line:
            model_path_line_index = i
            break
    
    if model_path_line_index is None:
        print("Error: MODEL_PATH not found in server.py")
        return False
    
    # Check if we already have MODEL_PATHS
    model_paths_line_index = None
    for i, line in enumerate(lines):
        if "MODEL_PATHS = " in line:
            model_paths_line_index = i
            break
    
    # Prepare the model paths content
    model_name = os.path.basename(model_path)
    model_name = os.path.splitext(model_name)[0]  # Remove extension
    
    # If we have MODEL_PATHS, update it
    if model_paths_line_index is not None:
        # Extract the current dictionary
        current_line = lines[model_paths_line_index].strip()
        # Check if the model is already in the dictionary
        if f"'{model_name}'" in current_line or f'"{model_name}"' in current_line:
            print(f"Model {model_name} is already in the MODEL_PATHS dictionary")
        else:
            # Find closing brace
            closing_brace_index = current_line.rfind("}")
            if closing_brace_index == -1:
                print("Error: Could not parse MODEL_PATHS dictionary")
                return False
            
            # Insert new model
            new_line = current_line[:closing_brace_index]
            # Add comma if not the first item
            if ":" in new_line:
                new_line += f", '{model_name}': r'{model_path}'"
            else:
                new_line += f"'{model_name}': r'{model_path}'"
            new_line += current_line[closing_brace_index:]
            lines[model_paths_line_index] = new_line + "\n"
    else:
        # Create MODEL_PATHS
        current_model_path = lines[model_path_line_index].strip()
        # Extract the path value from MODEL_PATH
        start_idx = current_model_path.find("=") + 1
        end_idx = len(current_model_path)
        if current_model_path.endswith("'") or current_model_path.endswith('"'):
            current_model_path_value = current_model_path[start_idx:end_idx].strip()
        else:
            # There might be a comment after the path
            if "#" in current_model_path:
                end_idx = current_model_path.find("#")
            current_model_path_value = current_model_path[start_idx:end_idx].strip()
        
        # Get the current model name
        current_model_name = "original"
        if os.path.exists(current_model_path_value.strip("'\"r ")):
            current_model_name = os.path.basename(current_model_path_value.strip("'\"r "))
            current_model_name = os.path.splitext(current_model_name)[0]
        
        # Create the MODEL_PATHS dictionary
        model_paths_line = f"MODEL_PATHS = {{'current': r{current_model_path_value}, '{current_model_name}': r{current_model_path_value}, '{model_name}': r'{model_path}'}}\n"
        
        # Insert the MODEL_PATHS line before MODEL_PATH
        lines.insert(model_path_line_index, model_paths_line)
        model_path_line_index += 1
        
        # Update MODEL_PATH to use the dictionary
        if is_default:
            lines[model_path_line_index] = f"MODEL_PATH = MODEL_PATHS['{model_name}']  # Default model\n"
        else:
            lines[model_path_line_index] = f"MODEL_PATH = MODEL_PATHS['current']  # Current active model\n"
    
    # Check if we need to add the model selection endpoint
    endpoint_exists = False
    for line in lines:
        if "@app.route('/select_model'" in line:
            endpoint_exists = True
            break
    
    if not endpoint_exists:
        # Find where to insert the endpoint
        insert_index = None
        for i, line in enumerate(lines):
            if "@app.route('/predict'" in line:
                # Find the end of this function
                for j in range(i, len(lines)):
                    if "return jsonify(" in lines[j] or "return json" in lines[j]:
                        insert_index = j + 1
                        while insert_index < len(lines) and lines[insert_index].strip() != "":
                            insert_index += 1
                        break
                break
        
        if insert_index is None:
            print("Warning: Could not find where to insert the model selection endpoint")
            insert_index = len(lines)  # Append at the end
        
        # Create model selection endpoint
        model_selection_endpoint = [
            "\n",
            "@app.route('/select_model', methods=['POST'])\n",
            "def select_model():\n",
            "    \"\"\"Endpoint to select which model to use for predictions\"\"\"\n",
            "    global MODEL_PATH\n",
            "    \n",
            "    # Get JSON data\n",
            "    data = request.json\n",
            "    model_name = data.get('model_name')\n",
            "    \n",
            "    if not model_name:\n",
            "        return jsonify({\n",
            "            'success': False,\n",
            "            'error': 'No model name provided',\n",
            "            'available_models': list(MODEL_PATHS.keys())\n",
            "        })\n",
            "    \n",
            "    if model_name not in MODEL_PATHS:\n",
            "        return jsonify({\n",
            "            'success': False,\n",
            "            'error': f'Model {model_name} not found',\n",
            "            'available_models': list(MODEL_PATHS.keys())\n",
            "        })\n",
            "    \n",
            "    # Update model path\n",
            "    MODEL_PATH = MODEL_PATHS[model_name]\n",
            "    \n",
            "    # Update 'current' model\n",
            "    MODEL_PATHS['current'] = MODEL_PATH\n",
            "    \n",
            "    # Reload model\n",
            "    load_model()\n",
            "    \n",
            "    return jsonify({\n",
            "        'success': True,\n",
            "        'message': f'Model switched to {model_name}',\n",
            "        'model_path': MODEL_PATH\n",
            "    })\n",
            "\n",
            "@app.route('/list_models', methods=['GET'])\n",
            "def list_models():\n",
            "    \"\"\"Endpoint to list available models\"\"\"\n",
            "    current_model = None\n",
            "    for name, path in MODEL_PATHS.items():\n",
            "        if path == MODEL_PATH:\n",
            "            current_model = name\n",
            "            break\n",
            "    \n",
            "    return jsonify({\n",
            "        'success': True,\n",
            "        'available_models': list(MODEL_PATHS.keys()),\n",
            "        'current_model': current_model\n",
            "    })\n",
        ]
        
        lines[insert_index:insert_index] = model_selection_endpoint
    
    # Update the load_model function to handle model changes
    load_model_start = None
    load_model_end = None
    for i, line in enumerate(lines):
        if "def load_model(" in line:
            load_model_start = i
            # Find the end of this function
            indentation = len(line) - len(line.lstrip())
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == "" or (len(lines[j]) - len(lines[j].lstrip())) <= indentation and lines[j].strip() != "":
                    load_model_end = j
                    break
            if load_model_end is None:
                load_model_end = len(lines)
            break
    
    if load_model_start is not None:
        # Check if we need to add global MODEL_PATH
        has_global = False
        for i in range(load_model_start, load_model_end):
            if "global MODEL_PATH" in lines[i]:
                has_global = True
                break
        
        if not has_global:
            # Find where to add the global declaration
            # Usually after the docstring or function def line
            insert_at = load_model_start + 1
            while insert_at < load_model_end and ('"""' in lines[insert_at] or "'''" in lines[insert_at]):
                insert_at += 1
            
            # Insert global declaration
            indent = " " * 4  # Assuming 4 spaces indentation
            lines.insert(insert_at, f"{indent}global MODEL_PATH, model\n")
            load_model_end += 1  # Increment end position due to inserted line
    
    # Write updates back to server file
    with open(server_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Server updated with new model: {model_name}")
    if is_default:
        print(f"Set {model_name} as the default model")
    
    return True

def modify_server_predict_route(server_path, add_debug_info=True):
    """
    Modifies the predict route to include model information and debug output
    """
    if not os.path.exists(server_path):
        print(f"Error: Server file not found at {server_path}")
        return False
    
    # Read the current server file
    with open(server_path, 'r') as f:
        content = f.read()
    
    # Find the predict route
    predict_route_start = content.find("@app.route('/predict'")
    if predict_route_start == -1:
        print("Error: Could not find predict route in server.py")
        return False
    
    # Find the return statement in the predict route
    return_start = content.find("return jsonify(", predict_route_start)
    if return_start == -1:
        print("Error: Could not find return statement in predict route")
        return False
    
    # Find the opening brace of the return JSON
    brace_start = content.find("{", return_start)
    if brace_start == -1:
        print("Error: Could not find JSON structure in return statement")
        return False
    
    # Find the closing brace of the return JSON
    brace_end = content.find("}", brace_start)
    if brace_end == -1:
        print("Error: Could not find end of JSON structure")
        return False
    
    # Get the current return JSON content
    return_json = content[brace_start:brace_end+1]
    
    # Prepare the updated return JSON (with model info)
    # Add model_info before the last closing brace
    last_comma_pos = return_json.rfind(",")
    last_brace_pos = return_json.rfind("}")
    
    if last_comma_pos > 0 and last_comma_pos > last_brace_pos - 10:
        # There's already a comma near the end
        insert_pos = last_brace_pos
    else:
        # Need to add a comma
        insert_pos = last_brace_pos
        return_json = return_json[:insert_pos] + "," + return_json[insert_pos:]
        insert_pos += 1
    
    # Add model info and debug info
    if add_debug_info:
        model_info = """
        'model_info': {
            'path': MODEL_PATH,
            # Extract model name from path
            'name': os.path.basename(MODEL_PATH).split('.')[0]
        },
        'debug_info': {
            'pixel_range': f"[{np.min(img_array):.2f}, {np.max(img_array):.2f}]",
            'mean_pixel_value': f"{np.mean(img_array):.4f}",
            'image_shape': str(img_array.shape)
        }"""
    else:
        model_info = """
        'model_info': {
            'path': MODEL_PATH,
            'name': os.path.basename(MODEL_PATH).split('.')[0]
        }"""
    
    updated_return_json = return_json[:insert_pos] + model_info + return_json[insert_pos:]
    
    # Replace the original return JSON with the updated one
    updated_content = content[:brace_start] + updated_return_json + content[brace_end+1:]
    
    # Update imports if needed
    if "import os" not in updated_content:
        # Find the last import
        last_import = 0
        for imp in ["import ", "from "]:
            last_pos = 0
            while True:
                pos = updated_content.find(imp, last_pos)
                if pos == -1:
                    break
                # Make sure this is a new line
                if pos == 0 or updated_content[pos-1] == '\n':
                    last_import = max(last_import, pos)
                last_pos = pos + 1
        
        # Add os import after the last import
        if last_import > 0:
            next_line = updated_content.find('\n', last_import)
            if next_line > 0:
                updated_content = updated_content[:next_line+1] + "import os\n" + updated_content[next_line+1:]
    
    # Write the updated content back to the file
    with open(server_path, 'w') as f:
        f.write(updated_content)
    
    print("Predict route updated with model information")
    if add_debug_info:
        print("Added debug information to prediction output")
    
    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Update server.py with new model')
    parser.add_argument('--server', type=str, default='server.py',
                      help='Path to the server.py file')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to the new model file')
    parser.add_argument('--set-default', action='store_true',
                      help='Set the new model as the default model')
    parser.add_argument('--add-debug', action='store_true',
                      help='Add debug information to prediction output')
    
    args = parser.parse_args()
    
    # Check if the model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return
    
    # Update the server with the new model
    success = update_server_with_new_model(args.server, args.model, args.set_default)
    
    if success and args.add_debug:
        # Modify the predict route to include debug information
        modify_server_predict_route(args.server, True)
    
    if success:
        print("\nServer updated successfully!")
        print("\nNext steps:")
        print("1. Restart the server: python server.py")
        print("2. Use the new endpoints to select and manage models:")
        print("   - GET /list_models - List all available models")
        print("   - POST /select_model - Switch to a different model")
        print("3. Test the new model on your problematic examples")

if __name__ == "__main__":
    main() 