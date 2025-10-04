########################
# Calculator REPL       #
########################

from decimal import Decimal
import logging
import ast
import operator as _op

from app.calculator import Calculator
from app.exceptions import OperationError, ValidationError
from app.history import AutoSaveObserver, LoggingObserver
from app.operations import OperationFactory
from app.input_validators import InputValidator
from app.calculator_memento import CalculatorMemento
from app.calculation import Calculation

# Allowed AST operator mapping
_ALLOWED_OPERATORS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.Pow: _op.pow,
    ast.UAdd: lambda x: x,
    ast.USub: _op.neg,
}

def _eval_expr(expr: str):
    """
    Safely evaluate a numeric expression using AST.
    Supports: +, -, *, /, ** (we'll map ^ -> ** before calling this),
    parentheses, and unary +/-. Does NOT allow function calls or names.
    Returns a Python number (int/float).
    """
    expr = expr.replace('^', '**')  # allow ^ as convenience

    node = ast.parse(expr, mode='eval')

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):  # Python 3.8+
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Unsupported constant")
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            func = _ALLOWED_OPERATORS.get(op_type)
            if func is None:
                raise ValueError("Operator not allowed")
            left = _eval(node.left)
            right = _eval(node.right)
            return func(left, right)
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            func = _ALLOWED_OPERATORS.get(op_type)
            if func is None:
                raise ValueError("Operator not allowed")
            return func(_eval(node.operand))
        raise ValueError("Unsupported expression element")

    return _eval(node.body)


def calculator_repl():
    """
    Command-line interface for the calculator.

    Implements a Read-Eval-Print Loop (REPL) that continuously prompts the user
    for commands, processes arithmetic operations, and manages calculation history.
    """
    try:
        # Initialize the Calculator instance
        calc = Calculator()

        # Register observers for logging and auto-saving history
        calc.add_observer(LoggingObserver())
        calc.add_observer(AutoSaveObserver(calc))

        print("Calculator started. Type 'help' for commands.")

        while True:
            try:
                # Prompt the user for a command
                raw = input("\nEnter command: ").strip()
                if not raw:
                    continue

                parts = raw.split(maxsplit=1)
                command = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else None

                if command == 'help':
                    # Display available commands
                    print("\nAvailable commands:")
                    print("  add (+), subtract (-), multiply (*), divide (/), power (^), root - Perform calculations")
                    print("  history - Show calculation history")
                    print("  clear - Clear calculation history")
                    print("  undo - Undo the last calculation")
                    print("  redo - Redo the last undone calculation")
                    print("  save - Save calculation history to file")
                    print("  load - Load calculation history from file")
                    print("  exit - Exit the calculator")
                    continue

                if command == 'exit':
                    # Attempt to save history before exiting
                    try:
                        calc.save_history()
                        print("History saved successfully.")
                    except Exception as e:
                        print(f"Warning: Could not save history: {e}")
                    print("Goodbye!")
                    break

                if command == 'history':
                    # Display calculation history
                    history = calc.show_history()
                    if not history:
                        print("No calculations in history")
                    else:
                        print("\nCalculation History:")
                        for i, entry in enumerate(history, 1):
                            print(f"{i}. {entry}")
                    continue

                if command == 'clear':
                    # Clear calculation history
                    calc.clear_history()
                    print("History cleared")
                    continue

                if command == 'undo':
                    # Undo the last calculation
                    if calc.undo():
                        print("Operation undone")
                    else:
                        print("Nothing to undo")
                    continue

                if command == 'redo':
                    # Redo the last undone calculation
                    if calc.redo():
                        print("Operation redone")
                    else:
                        print("Nothing to redo")
                    continue

                if command == 'save':
                    # Save calculation history to file
                    try:
                        calc.save_history()
                        print("History saved successfully")
                    except Exception as e:
                        print(f"Error saving history: {e}")
                    continue

                if command == 'load':
                    # Load calculation history from file
                    try:
                        calc.load_history()
                        print("History loaded successfully")
                    except Exception as e:
                        print(f"Error loading history: {e}")
                    continue

                # ---------- expr handling ----------
                if command == 'expr':
                    expr_input = arg or input("Enter expression: ").strip()
                    try:
                        raw_value = _eval_expr(expr_input)
                        # convert to Decimal and validate via InputValidator
                        validated_result = InputValidator.validate_number(str(raw_value), calc.config)

                        # Save memento & append to history (keep undo/redo behavior consistent)
                        calc.undo_stack.append(CalculatorMemento(calc.history.copy()))
                        calc.redo_stack.clear()

                        # Create a Calculation entry where operation is "Expression"
                        calc_entry = Calculation(operation="Expression",
                                                 operand1=validated_result,
                                                 operand2=Decimal('0'))
                        calc.history.append(calc_entry)

                        # Notify observers (auto-save/logging)
                        calc.notify_observers(calc_entry)

                        print(f"\nResult: {validated_result}")
                    except (ValueError, ValidationError) as e:
                        print(f"Invalid expression: {e}")
                    except Exception as e:
                        print(f"Unexpected error evaluating expression: {e}")
                    continue

                # ---------- arithmetic (support word and symbol aliases) ----------
                if command in ['add', '+', 'subtract', '-', 'multiply', '*', 'divide', '/', 'power', '^', 'root']:
                    # Perform the specified arithmetic operation
                    try:
                        print("\nEnter numbers (or 'cancel' to abort):")
                        a = input("First number: ")
                        if a.lower() == 'cancel':
                            print("Operation cancelled")
                            continue
                        b = input("Second number: ")
                        if b.lower() == 'cancel':
                            print("Operation cancelled")
                            continue

                        # Create the appropriate operation instance using the Factory pattern
                        operation = OperationFactory.create_operation(command)
                        calc.set_operation(operation)

                        # Perform the calculation
                        result = calc.perform_operation(a, b)

                        # Normalize the result if it's a Decimal
                        if isinstance(result, Decimal):
                            result = result.normalize()

                        print(f"\nResult: {result}")
                    except (ValidationError, OperationError) as e:
                        # Handle known exceptions related to validation or operation errors
                        print(f"Error: {e}")
                    except Exception as e:
                        # Handle any unexpected exceptions
                        print(f"Unexpected error: {e}")
                    continue

                # Handle unknown commands
                print(f"Unknown command: '{command}'. Type 'help' for available commands.")

            except KeyboardInterrupt:
                # Handle Ctrl+C interruption gracefully
                print("\nOperation cancelled")
                continue
            except EOFError:
                # Handle end-of-file (e.g., Ctrl+D) gracefully
                print("\nInput terminated. Exiting...")
                break
            except Exception as e:
                # Handle any other unexpected exceptions
                print(f"Error: {e}")
                continue

    except Exception as e:
        # Handle fatal errors during initialization
        print(f"Fatal error: {e}")
        logging.error(f"Fatal error in calculator REPL: {e}")
        raise
