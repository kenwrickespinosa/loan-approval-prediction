<?php

namespace App\Http\Controllers;

use App\Models\Client;
use App\Models\Loan;
use Illuminate\Http\Request;

class LoanController extends Controller
{
    public function index()
    {
        return Loan::with('client')->latest()->get();
    }

    public function store(Request $request)
    {
        $validated = $request->validate([
            'client_id' => 'nullable|exists:clients,id',
            'gender' => 'nullable|string',
            'married' => 'nullable|string',
            'dependents' => 'nullable|string',
            'education' => 'nullable|string',
            'self_employed' => 'nullable|string',
            'applicant_income' => 'nullable|integer',
            'coapplicant_income' => 'nullable|integer',
            'loan_amount' => 'nullable|integer',
            'loan_amount_term' => 'nullable|integer',
            'credit_history' => 'nullable|string',
            'property_area' => 'nullable|string',
            'loan_status' => 'required|string',
        ]);

        $loan = Loan::create($validated);

        return response()->json([
            'message' => 'Loan evaluation saved successfully',
            'data' => $loan
        ], 201);
    }

    public function totalAmountReq(Request $request)
    {
        $query = Loan::query();

        if ($request->filled('married') && $request->married !== 'all') {
            $query->where('married', $request->married);
        }

        if ($request->filled('gender') && $request->gender !== 'all') {
            $query->whereHas('client', function ($q) use ($request) {
                $q->where('gender', $request->gender);
            });
        }

        return response()->json([
            'data' => $query->sum('loan_amount'),
        ]);
    }

    public function totalLoanStatus(Request $request)
    {
        $user = $request->user();

        $query = Loan::whereHas('client', function ($q) use ($user) {
            $q->where('user_id', $user->id);
        });

        if ($request->filled('gender') && $request->gender !== 'all') {
            $query->whereHas('client', function ($q) use ($request) {
                $q->where('gender', $request->gender);
            });
        }

        if ($request->filled('married') && $request->married != 'all') {
            $query->where('married', $request->married);
        }

        return response()->json([
            'approved' => (clone $query)->where('loan_status', 'Y')->count(),
            'rejected' => (clone $query)->where('loan_status', 'N')->count(),
        ]);
    }
}
