let expressionDisplay = document.getElementById('expression');
let resultDisplay = document.getElementById('result');
let currentExpression = '';

// Function to append characters to the expression
function appendToExpression(value) {
    // Prevent multiple operators in a row (except minus for negative numbers)
    const operators = ['+', '-', '*', '/', '%'];
    const lastChar = currentExpression.slice(-1);
    
    if (operators.includes(value) && operators.includes(lastChar) && value !== '-') {
        // Replace the last operator with the new one
        currentExpression = currentExpression.slice(0, -1) + value;
    } else if (value === '.' && lastChar === '.') {
        // Prevent multiple decimal points in a row
        return;
    } else {
        // Append the value to the expression
        currentExpression += value;
    }
    
    // Update the display
    expressionDisplay.textContent = currentExpression;
}

// Function to clear the display
function clearDisplay() {
    currentExpression = '';
    expressionDisplay.textContent = '';
    resultDisplay.textContent = '0';
}

// Function to remove the last character (backspace)
function backspace() {
    currentExpression = currentExpression.slice(0, -1);
    expressionDisplay.textContent = currentExpression;
    
    if (currentExpression === '') {
        resultDisplay.textContent = '0';
    }
}

// Function to calculate the result
function calculate() {
    if (currentExpression === '') {
        return;
    }
    
    try {
        // Send the expression to the server for calculation
        fetch('/calculate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ expression: currentExpression }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                resultDisplay.textContent = 'Error';
                console.error(data.error);
            } else {
                resultDisplay.textContent = data.result;
                currentExpression = data.result.toString();
                expressionDisplay.textContent = '';
            }
        })
        .catch(error => {
            resultDisplay.textContent = 'Error';
            console.error('Error:', error);
        });
    } catch (error) {
        resultDisplay.textContent = 'Error';
        console.error('Error:', error);
    }
}

// Add keyboard support
document.addEventListener('keydown', function(event) {
    const key = event.key;
    
    if (key >= '0' && key <= '9') {
        appendToExpression(key);
    } else if (key === '+' || key === '-' || key === '*' || key === '/' || key === '%' || key === '.') {
        appendToExpression(key);
    } else if (key === 'Enter' || key === '=') {
        calculate();
    } else if (key === 'Backspace') {
        backspace();
    } else if (key === 'Escape' || key === 'Delete') {
        clearDisplay();
    }
});
