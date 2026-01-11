#!/usr/bin/env node
/**
 * Babel-based AST analyzer for JavaScript/TypeScript files.
 * Outputs JSON metrics to stdout.
 * 
 * Usage: node analyze_js_ast.js <filepath>
 */

const fs = require('fs');
const path = require('path');
const parser = require('@babel/parser');
const traverse = require('@babel/traverse').default;

function analyzeFile(filePath) {
    const metrics = {
        loc: 0,
        cc: 1,  // Base cyclomatic complexity
        mi: null,
        function_count: 0,
        import_count: 0
    };

    let code;
    try {
        code = fs.readFileSync(filePath, 'utf-8');
    } catch (err) {
        console.error(JSON.stringify({ error: `Cannot read file: ${err.message}` }));
        process.exit(1);
    }

    // Determine plugins based on file extension
    const ext = path.extname(filePath).toLowerCase();
    const plugins = ['decorators-legacy', 'classProperties', 'classPrivateProperties', 'classPrivateMethods'];
    
    if (ext === '.tsx' || ext === '.ts') {
        plugins.push('typescript');
    }
    if (ext === '.jsx' || ext === '.tsx') {
        plugins.push('jsx');
    }
    if (ext === '.js' || ext === '.jsx') {
        plugins.push('jsx');  // Many .js files use JSX in React projects
    }

    let ast;
    try {
        ast = parser.parse(code, {
            sourceType: 'unambiguous',  // Detect module vs script
            plugins: plugins,
            errorRecovery: true  // Continue parsing even with errors
        });
    } catch (err) {
        console.error(JSON.stringify({ error: `Parse error: ${err.message}` }));
        process.exit(1);
    }

    // Count LOC (non-empty, non-comment lines)
    const lines = code.split('\n');
    let loc = 0;
    let inBlockComment = false;
    
    for (const line of lines) {
        const trimmed = line.trim();
        
        // Handle block comments
        if (trimmed.includes('/*') && trimmed.includes('*/')) {
            // Single-line block comment - check if there's code around it
            const withoutComment = trimmed.replace(/\/\*.*?\*\//g, '').trim();
            if (withoutComment && !withoutComment.startsWith('//')) {
                loc++;
            }
            continue;
        }
        if (trimmed.includes('/*')) {
            inBlockComment = true;
            // Check if there's code before the comment start
            const beforeComment = trimmed.split('/*')[0].trim();
            if (beforeComment) loc++;
            continue;
        }
        if (trimmed.includes('*/')) {
            inBlockComment = false;
            // Check if there's code after the comment end
            const afterComment = trimmed.split('*/')[1]?.trim();
            if (afterComment && !afterComment.startsWith('//')) loc++;
            continue;
        }
        
        if (inBlockComment) continue;
        
        // Skip empty lines and single-line comments
        if (trimmed && !trimmed.startsWith('//')) {
            loc++;
        }
    }
    metrics.loc = loc;

    // Traverse AST to count metrics
    let ccTotal = 1;  // Base complexity
    
    traverse(ast, {
        // Function definitions
        FunctionDeclaration() {
            metrics.function_count++;
        },
        FunctionExpression() {
            metrics.function_count++;
        },
        ArrowFunctionExpression() {
            metrics.function_count++;
        },
        ClassMethod() {
            metrics.function_count++;
        },
        ObjectMethod() {
            metrics.function_count++;
        },

        // Import statements
        ImportDeclaration() {
            metrics.import_count++;
        },
        // CommonJS require()
        CallExpression(path) {
            if (path.node.callee.name === 'require' && path.node.arguments.length > 0) {
                metrics.import_count++;
            }
        },

        // Cyclomatic complexity decision points
        IfStatement() {
            ccTotal++;
        },
        ForStatement() {
            ccTotal++;
        },
        ForInStatement() {
            ccTotal++;
        },
        ForOfStatement() {
            ccTotal++;
        },
        WhileStatement() {
            ccTotal++;
        },
        DoWhileStatement() {
            ccTotal++;
        },
        SwitchCase(path) {
            // Don't count default case
            if (path.node.test !== null) {
                ccTotal++;
            }
        },
        ConditionalExpression() {
            ccTotal++;  // Ternary operator
        },
        LogicalExpression(path) {
            // && and || add to complexity
            if (path.node.operator === '&&' || path.node.operator === '||') {
                ccTotal++;
            }
        },
        CatchClause() {
            ccTotal++;
        },
        OptionalMemberExpression() {
            ccTotal++;  // ?. optional chaining
        },
        OptionalCallExpression() {
            ccTotal++;  // ?.() optional call
        }
    });

    // Calculate average CC per function, or total if no functions
    if (metrics.function_count > 0) {
        metrics.cc = Math.round((ccTotal / metrics.function_count) * 100) / 100;
    } else {
        metrics.cc = ccTotal;
    }

    return metrics;
}

// Main
const args = process.argv.slice(2);
if (args.length === 0) {
    console.error(JSON.stringify({ error: 'Usage: node analyze_js_ast.js <filepath>' }));
    process.exit(1);
}

const filePath = args[0];
if (!fs.existsSync(filePath)) {
    console.error(JSON.stringify({ error: `File not found: ${filePath}` }));
    process.exit(1);
}

const metrics = analyzeFile(filePath);
console.log(JSON.stringify(metrics));

