<?php

use App\Http\Controllers\AuthController;
use App\Http\Controllers\ClientController;
use App\Http\Controllers\LoanController;
use App\Http\Controllers\PredictController;
use Illuminate\Support\Facades\Route;

Route::post('/login', [AuthController::class, 'login']);
Route::post('/register', [AuthController::class, 'register']);
Route::middleware('auth:sanctum')->post('/logout', [AuthController::class, 'logout']);
Route::middleware('auth:sanctum')->group(function() {
    Route::get('/clients', [ClientController::class, 'index']);
    Route::post('/clients', [ClientController::class, 'store']);
    Route::get('/clients/count', [ClientController::class, 'count']);
    Route::get('/clients/clients-with-and-without-loans', [ClientController::class, 'clientsWithAndWithoutLoans']);
});

// Route::middleware('auth:sanctum')->post('/predict-loan', [PredictController::class, 'predict']);
Route::post('/predict-loan', [PredictController::class, 'predict']);

Route::post('/loan', [LoanController::class, 'store']);
Route::middleware('auth:sanctum')->get('/loan/total-amount-requested', [LoanController::class, 'totalAmountReq']);
Route::middleware('auth:sanctum')->get('/loan/total-loan-status', [LoanController::class, 'totalLoanStatus']);