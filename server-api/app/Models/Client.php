<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\HasMany;

class Client extends Model
{
    protected $fillable = [
        'user_id',
        'firstname',
        'lastname',
        'gender',
        'birthdate',
        'address',
        'contact_number'
    ];

    public function user() {
        return $this->belongsTo(User::class);
    }

    public function loans(): HasMany {
        return $this->hasMany(Loan::class);
    }
}
