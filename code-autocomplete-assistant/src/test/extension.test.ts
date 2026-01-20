import * as assert from 'assert';

// You can import and use all API from the 'vscode' module
// as well as import your extension to test it
import * as vscode from 'vscode';
// import * as myExtension from '../../extension';

function fibbonaci(n: int) {

    if (n <= 1) {
        return n;
    }
    
    return fibbonaci(n - 1) + fibbonaci(n - 2);  // tail call optimization is not supported in nodejs 10.x
    // return fibbonaci(n - 1) +

}