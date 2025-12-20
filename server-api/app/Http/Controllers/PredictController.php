<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

class PredictController extends Controller
{
    public function predict(Request $request)
    {
        $input = [
            "Gender" => $request->gender === "Male" ? 1 : 0,
            "Married" => $request->married === "Yes" ? 1 : 0,
            "Dependents" => min(max(intval($request->dependents), 0), 3),
            "Education" => $request->education === "Graduate" ? 1 : 0,
            "Self_Employed" => $request->self_employed === "Yes" ? 1 : 0,
            "ApplicantIncome" => floatval($request->applicant_income),
            "CoapplicantIncome" => floatval($request->coapplicant_income),
            "LoanAmount" => floatval($request->loan_amount),
            "Loan_Amount_Term" => floatval($request->loan_amount_term),
            "Credit_History" => $request->credit_history === "Yes" ? 1 : 0,
            "Property_Area" => match ($request->property_area) {
                "Rural" => 0,
                "Urban" => 1,
                "Semi-Urban" => 2
            },
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
        fclose($pipes[0]);

        $result = stream_get_contents($pipes[1]);
        fclose($pipes[1]);

        $error = stream_get_contents($pipes[2]);
        fclose($pipes[2]);

        proc_close($process);

        if (!empty($error)) {
            return response()->json(["error" => $error], 500);
        }

        return response()->json(json_decode($result, true));
        // return response()->json([
        //     "prediction" => $result["Prediction"]
        // ]);
    }
}
