<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Loan extends Model
{
    protected $fillable = [
        'client_id',
        'gender',
        'married',
        'dependents',
        'education',
        'self_employed',
        'applicant_income',
        'coapplicant_income',
        'loan_amount',
        'loan_amount_term',
        'credit_history',
        'property_area',
        'loan_status'
    ];

    public function client() {
        return $this->belongsTo(Client::class);
    }
}
