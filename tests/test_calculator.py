import datetime
from pathlib import Path
import pandas as pd
import pytest
from unittest.mock import Mock, patch, PropertyMock
from decimal import Decimal
from tempfile import TemporaryDirectory
from app.calculation import Calculation
from app.calculator import Calculator
from app.calculator_repl import calculator_repl, _eval_expr
from app.calculator_memento import CalculatorMemento
from app.input_validators import InputValidator
from app.calculator_config import CalculatorConfig
from app.exceptions import OperationError, ValidationError
from app.history import LoggingObserver, AutoSaveObserver
from app.operations import OperationFactory


# Fixture to initialize Calculator with a temporary directory for file paths
@pytest.fixture
def calculator():
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = CalculatorConfig(base_dir=temp_path)

        # Patch properties to use the temporary directory paths
        with patch.object(CalculatorConfig, 'log_dir', new_callable=PropertyMock) as mock_log_dir, \
             patch.object(CalculatorConfig, 'log_file', new_callable=PropertyMock) as mock_log_file, \
             patch.object(CalculatorConfig, 'history_dir', new_callable=PropertyMock) as mock_history_dir, \
             patch.object(CalculatorConfig, 'history_file', new_callable=PropertyMock) as mock_history_file:
            
            # Set return values to use paths within the temporary directory
            mock_log_dir.return_value = temp_path / "logs"
            mock_log_file.return_value = temp_path / "logs/calculator.log"
            mock_history_dir.return_value = temp_path / "history"
            mock_history_file.return_value = temp_path / "history/calculator_history.csv"
            
            # Return an instance of Calculator with the mocked config
            yield Calculator(config=config)

# Test Calculator Initialization

def test_calculator_initialization(calculator):
    assert calculator.history == []
    assert calculator.undo_stack == []
    assert calculator.redo_stack == []
    assert calculator.operation_strategy is None

# Test Logging Setup

@patch('app.calculator.logging.info')
def test_logging_setup(logging_info_mock):
    with patch.object(CalculatorConfig, 'log_dir', new_callable=PropertyMock) as mock_log_dir, \
         patch.object(CalculatorConfig, 'log_file', new_callable=PropertyMock) as mock_log_file:
        mock_log_dir.return_value = Path('/tmp/logs')
        mock_log_file.return_value = Path('/tmp/logs/calculator.log')
        
        # Instantiate calculator to trigger logging
        calculator = Calculator(CalculatorConfig())
        logging_info_mock.assert_any_call("Calculator initialized with configuration")

# Test Adding and Removing Observers

def test_add_observer(calculator):
    observer = LoggingObserver()
    calculator.add_observer(observer)
    assert observer in calculator.observers

def test_remove_observer(calculator):
    observer = LoggingObserver()
    calculator.add_observer(observer)
    calculator.remove_observer(observer)
    assert observer not in calculator.observers

# Test Setting Operations

def test_set_operation(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    assert calculator.operation_strategy == operation

# Test Performing Operations

def test_perform_operation_addition(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    result = calculator.perform_operation(2, 3)
    assert result == Decimal('5')

def test_perform_operation_validation_error(calculator):
    calculator.set_operation(OperationFactory.create_operation('add'))
    with pytest.raises(ValidationError):
        calculator.perform_operation('invalid', 3)

def test_perform_operation_operation_error(calculator):
    with pytest.raises(OperationError, match="No operation set"):
        calculator.perform_operation(2, 3)

# Test Undo/Redo Functionality

def test_undo(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.undo()
    assert calculator.history == []

def test_redo(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.undo()
    calculator.redo()
    assert len(calculator.history) == 1

# Test History Management

@patch('app.calculator.pd.DataFrame.to_csv')
def test_save_history(mock_to_csv, calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.save_history()
    mock_to_csv.assert_called_once()

@patch('app.calculator.pd.read_csv')
@patch('app.calculator.Path.exists', return_value=True)
def test_load_history(mock_exists, mock_read_csv, calculator):
    # Mock CSV data to match the expected format in from_dict
    mock_read_csv.return_value = pd.DataFrame({
        'operation': ['Addition'],
        'operand1': ['2'],
        'operand2': ['3'],
        'result': ['5'],
        'timestamp': [datetime.datetime.now().isoformat()]
    })
    
    # Test the load_history functionality
    try:
        calculator.load_history()
        # Verify history length after loading
        assert len(calculator.history) == 1
        # Verify the loaded values
        assert calculator.history[0].operation == "Addition"
        assert calculator.history[0].operand1 == Decimal("2")
        assert calculator.history[0].operand2 == Decimal("3")
        assert calculator.history[0].result == Decimal("5")
    except OperationError:
        pytest.fail("Loading history failed due to OperationError")
        
            
# Test Clearing History

def test_clear_history(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.clear_history()
    assert calculator.history == []
    assert calculator.undo_stack == []
    assert calculator.redo_stack == []

# Test REPL Commands (using patches for input/output handling)

@patch('builtins.input', side_effect=['exit'])
@patch('builtins.print')
def test_calculator_repl_exit(mock_print, mock_input):
    with patch('app.calculator.Calculator.save_history') as mock_save_history:
        calculator_repl()
        mock_save_history.assert_called_once()
        mock_print.assert_any_call("History saved successfully.")
        mock_print.assert_any_call("Goodbye!")

@patch('builtins.input', side_effect=['help', 'exit'])
@patch('builtins.print')
def test_calculator_repl_help(mock_print, mock_input):
    calculator_repl()
    mock_print.assert_any_call("\nAvailable commands:")

@patch('builtins.input', side_effect=['add', '2', '3', 'exit'])
@patch('builtins.print')
def test_calculator_repl_addition(mock_print, mock_input):
    calculator_repl()
    mock_print.assert_any_call("\nResult: 5")

@patch('builtins.input', side_effect=['exit'])
@patch('builtins.print')
@patch('app.calculator.Calculator.save_history', side_effect=Exception("Save failed"))
def test_calculator_repl_exit_save_error(mock_save, mock_print, mock_input):
    """Test that exit handles save errors gracefully"""
    calculator_repl()
    # Check that warning was printed
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("Warning: Could not save history" in call for call in calls)
    assert any("Goodbye!" in call for call in calls)


# Test history command with entries
@patch('builtins.input', side_effect=['add', '5', '3', 'history', 'exit'])
@patch('builtins.print')
def test_calculator_repl_history_with_entries(mock_print, mock_input):
    """Test history command displays calculation entries"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("Calculation History:" in call for call in calls)

# Test clear command
@patch('builtins.input', side_effect=['add', '5', '3', 'clear', 'exit'])
@patch('builtins.print')
def test_calculator_repl_clear(mock_print, mock_input):
    """Test clear command clears history"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("History cleared" in call for call in calls)


# Test undo when there's something to undo
@patch('builtins.input', side_effect=['add', '5', '3', 'undo', 'exit'])
@patch('builtins.print')
def test_calculator_repl_undo_success(mock_print, mock_input):
    """Test undo command successfully undoes operation"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("Operation undone" in call for call in calls)


# Test undo when nothing to undo
@patch('builtins.input', side_effect=['undo', 'exit'])
@patch('builtins.print')
def test_calculator_repl_undo_nothing(mock_print, mock_input):
    """Test undo command when nothing to undo"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("Nothing to undo" in call for call in calls)


# Test redo when there's something to redo
@patch('builtins.input', side_effect=['add', '5', '3', 'undo', 'redo', 'exit'])
@patch('builtins.print')
def test_calculator_repl_redo_success(mock_print, mock_input):
    """Test redo command successfully redoes operation"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("Operation redone" in call for call in calls)


# Test redo when nothing to redo
@patch('builtins.input', side_effect=['redo', 'exit'])
@patch('builtins.print')
def test_calculator_repl_redo_nothing(mock_print, mock_input):
    """Test redo command when nothing to redo"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("Nothing to redo" in call for call in calls)


# Test save command success
@patch('builtins.input', side_effect=['save', 'exit'])
@patch('builtins.print')
def test_calculator_repl_save_success(mock_print, mock_input):
    """Test save command saves successfully"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("History saved successfully" in call for call in calls)


# Test save command error
@patch('builtins.input', side_effect=['save', 'exit'])
@patch('builtins.print')
@patch('app.calculator.Calculator.save_history', side_effect=Exception("Save error"))
def test_calculator_repl_save_error(mock_save, mock_print, mock_input):
    """Test save command handles errors"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("Error saving history" in call for call in calls)


# Test load command success
@patch('builtins.input', side_effect=['load', 'exit'])
@patch('builtins.print')
def test_calculator_repl_load_success(mock_print, mock_input):
    """Test load command loads successfully"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("History loaded successfully" in call for call in calls)


# Test load command error
@patch('builtins.input', side_effect=['load', 'exit'])
@patch('builtins.print')
@patch('app.calculator.Calculator.load_history', side_effect=Exception("Load error"))
def test_calculator_repl_load_error(mock_load, mock_print, mock_input):
    """Test load command handles errors"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("Error loading history" in call for call in calls)


# Test expr command with inline expression
@patch('builtins.input', side_effect=['expr 2 + 3', 'exit'])
@patch('builtins.print')
def test_calculator_repl_expr_inline(mock_print, mock_input):
    """Test expr command with inline expression"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("Result: 5" in call for call in calls)


# Test expr command with prompt
@patch('builtins.input', side_effect=['expr', '10 / 2', 'exit'])
@patch('builtins.print')
def test_calculator_repl_expr_prompt(mock_print, mock_input):
    """Test expr command with prompted expression"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("Result: 5" in call for call in calls)


# Test expr with validation error
@patch('builtins.input', side_effect=['expr', '2 + 3', 'exit'])
@patch('builtins.print')
@patch('app.input_validators.InputValidator.validate_number', 
       side_effect=ValidationError("Number too large"))
def test_calculator_repl_expr_validation_error(mock_validate, mock_print, mock_input):
    """Test expr command with validation error"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("Invalid expression" in call for call in calls)


# Test expr with unexpected error
@patch('builtins.input', side_effect=['expr', '5 + 5', 'exit'])
@patch('builtins.print')
@patch('app.calculator_repl._eval_expr', side_effect=RuntimeError("Unexpected"))
def test_calculator_repl_expr_unexpected_error(mock_eval, mock_print, mock_input):
    """Test expr command with unexpected error"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("Unexpected error evaluating expression" in call for call in calls)


# Test arithmetic operations with all aliases
@patch('builtins.input', side_effect=['subtract', '10', '3', 'exit'])
@patch('builtins.print')
def test_calculator_repl_subtract(mock_print, mock_input):
    """Test subtract command"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("Result: 7" in call for call in calls)

@patch('builtins.input', side_effect=['add', 'cancel', 'exit'])
@patch('builtins.print')
def test_calculator_repl_cancel_first(mock_print, mock_input):
    """Test canceling operation at first number"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("Operation cancelled" in call for call in calls)


# Test cancel at second number
@patch('builtins.input', side_effect=['add', '5', 'cancel', 'exit'])
@patch('builtins.print')
def test_calculator_repl_cancel_second(mock_print, mock_input):
    """Test canceling operation at second number"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("Operation cancelled" in call for call in calls)


# Test ValidationError in operation
@patch('builtins.input', side_effect=['add', 'not_a_number', '3', 'exit'])
@patch('builtins.print')
def test_calculator_repl_validation_error(mock_print, mock_input):
    """Test operation with invalid input"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("Error:" in call for call in calls)


# Test OperationError in operation
@patch('builtins.input', side_effect=['/', '10', '0', 'exit'])
@patch('builtins.print')
def test_calculator_repl_operation_error(mock_print, mock_input):
    """Test operation that causes OperationError (division by zero)"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("Error:" in call for call in calls)


# Test unexpected exception in operation
@patch('builtins.input', side_effect=['add', '5', '3', 'exit'])
@patch('builtins.print')
@patch('app.calculator.Calculator.perform_operation', side_effect=RuntimeError("Unexpected"))
def test_calculator_repl_unexpected_error(mock_perform, mock_print, mock_input):
    """Test unexpected error during operation"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("Unexpected error:" in call for call in calls)


# Test unknown command
@patch('builtins.input', side_effect=['unknown_cmd', 'exit'])
@patch('builtins.print')
def test_calculator_repl_unknown_command(mock_print, mock_input):
    """Test unknown command handling"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("Unknown command: 'unknown_cmd'" in call for call in calls)


# Test KeyboardInterrupt (Ctrl+C)
@patch('builtins.input', side_effect=[KeyboardInterrupt(), 'exit'])
@patch('builtins.print')
def test_calculator_repl_keyboard_interrupt(mock_print, mock_input):
    """Test handling of KeyboardInterrupt"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("Operation cancelled" in call for call in calls)


# Test EOFError
@patch('builtins.input', side_effect=EOFError())
@patch('builtins.print')
def test_calculator_repl_eof(mock_print, mock_input):
    """Test handling of EOFError"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("Input terminated" in call for call in calls)


# Test Exception in main loop
@patch('builtins.input', side_effect=[RuntimeError("Test error"), 'exit'])
@patch('builtins.print')
def test_calculator_repl_exception_in_loop(mock_print, mock_input):
    """Test handling of unexpected exception in main loop"""
    calculator_repl()
    calls = [str(call) for call in mock_print.call_args_list]
    assert any("Error:" in call for call in calls)

@patch('builtins.input', side_effect=['exit'])
@patch('app.calculator.Calculator.__init__', side_effect=Exception("Fatal init error"))
def test_repl_fatal_error_on_init(mock_init, mock_input, capsys):
    """Test the REPL's fatal error handler on calculator initialization failure."""
    with pytest.raises(Exception, match="Fatal init error"):
        calculator_repl()
    
    captured = capsys.readouterr()
    assert "Fatal error: Fatal init error" in captured.out


# expression evaluation tests for _eval_expr function
def test_eval_expr_basic_operations():
    assert _eval_expr("1+2") == 3
    assert _eval_expr("5 + 3 * 2") == 11
    assert _eval_expr("(2 + 3) ^ 2") == 25  # ^ is mapped to **
    assert _eval_expr("10 / 2") == 5

def test_eval_expr_constant_node():
    result = _eval_expr("42")
    assert result == 42

def test_eval_expr_unary_minus():
    assert _eval_expr("-5 + 2") == -3

def test_eval_expr_invalid():
    with pytest.raises(ValueError):
        _eval_expr("os.system('rm -rf /')")  # not allowed

def test_eval_expr_invalid_operator_binop():
    with pytest.raises(ValueError, match="Operator not allowed"):
        _eval_expr("5 % 2")

def test_eval_expr_invalid_operator_unary():
    with pytest.raises(ValueError, match="Operator not allowed"):
        _eval_expr("~5")

def test_eval_expr_unsupported_constant():
    with pytest.raises(ValueError, match="Unsupported constant"):
        _eval_expr("'hello'")

def test_eval_expr_unsupported_expression_element():
    with pytest.raises(ValueError, match="Unsupported expression element"):
        _eval_expr("some_var")

def test_expr_integration_with_calculator():
    calc = Calculator()
    # Simulate "expr 5 + 3 * 2"
    result = Decimal(str(_eval_expr("5 + 3 * 2")))
    validated = InputValidator.validate_number(str(result), calc.config)

    # Append manually as REPL would
    entry = Calculation(operation="Expression", operand1=validated, operand2=Decimal('0'))
    calc.history.append(entry)

    assert calc.history[-1].operation == "Expression"
    assert calc.history[-1].calculate() == validated

# Test error handling in Calculator methods
def test_setup_logging_fails(capsys):
    """Test that an error during logging setup is handled."""
    with patch('logging.basicConfig', side_effect=Exception("Permission denied")):
        with pytest.raises(Exception, match="Permission denied"):
            # We must instantiate the class inside the test to trigger the error
            Calculator(CalculatorConfig())
        # Check that the print statement was called
        captured = capsys.readouterr()
        assert "Error setting up logging" in captured.out

def test_perform_operation_generic_exception(calculator):
    """Test handling of a generic exception during an operation."""
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    
    # Mock the execute method to raise a generic Exception
    with patch.object(operation, 'execute', side_effect=Exception("A random error")):
        with pytest.raises(OperationError, match="Operation failed: A random error"):
            calculator.perform_operation(2, 3)

def test_save_history_fails(calculator):
    """Test that an error during history saving is handled."""
    with patch('pandas.DataFrame.to_csv', side_effect=Exception("Disk full")):
        with pytest.raises(OperationError, match="Failed to save history: Disk full"):
            calculator.save_history()

def test_load_history_fails(calculator):
    """Test that an error during history loading is handled."""
    # Ensure the history file exists for the test to attempt reading it
    calculator.config.history_file.touch()
    with patch('pandas.read_csv', side_effect=Exception("Corrupt file")):
        with pytest.raises(OperationError, match="Failed to load history: Corrupt file"):
            calculator.load_history()

@patch('logging.warning')
def test_init_load_history_fails(mock_warning):
    """Test that a failure to load history during __init__ is logged."""
    with patch.object(Calculator, 'load_history', side_effect=Exception("Cannot load")):
        Calculator()
        mock_warning.assert_called_once_with("Could not load existing history: Cannot load")

# Tests for Calculation memento
def test_memento_with_empty_history():
    """Test memento serialization with an empty history list."""
    memento = CalculatorMemento(history=[])
    memento_dict = memento.to_dict()

    # Test to_dict
    assert memento_dict['history'] == []
    
    # Test from_dict
    restored_memento = CalculatorMemento.from_dict(memento_dict)
    assert restored_memento.history == []