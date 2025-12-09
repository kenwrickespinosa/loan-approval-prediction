<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

class PredictController extends Controller
{
    public function predict(Request $request) {
        $input = [
            "Gender" => intval($request->gender),
            "Married" => intval($request->married),
            "Dependents" => intval($request->dependents),
            "Education" => intval($request->education),
            "Self_Employed" => intval($request->self_employed),
            "ApplicantIncome" => floatval($request->applicant_income),
            "CoapplicantIncome" => floatval($request->coapplicant_income),
            "LoanAmount" => floatval($request->loan_amount),
            "Loan_Amount_Term" => floatval($request->loan_amount_term),
            "Credit_History" => intval($request->credit_history),
            "Property_Area" => intval($request->property_area),
        ];

        $jsonInput = json_encode($input);

        $process = proc_open(
            'python predmodels/predict.py',
            [
                0 => ['pipe', 'r'],    // STDIN
                1 => ['pipe', 'w'],    // STDOUT
                2 => ['pipe', 'w'],    // STDERR
            ],
            $pipes,
            base_path()    // Run inside Laravel root
        );

        fwrite($pipes[0], $jsonInput);
        fclose($pipes[1]);

        $result = stream_get_contents($pipes[1]);
        fclose($pipes[1]);

        $error = stream_get_contents($pipes[2]);
        fclose($pipes[2]);

        proc_close($process);

        if (!empty($error)) {
            return response()->json(["error" => $error], 500);
        }

        return response()->json(json_decode($result, true));
    }
}
