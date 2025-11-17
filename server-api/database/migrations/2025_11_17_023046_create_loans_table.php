<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('loans', function (Blueprint $table) {
            $table->id();
            $table->foreignId('client_id')->constrained('clients')->nullOnDelete();
            $table->string('gender');
            $table->string('married');
            $table->integer('dependents');
            $table->string('education');
            $table->string('self_employed');
            $table->decimal('applicant_income', 10, 2);
            $table->decimal('coapplicant_income', 10, 2);
            $table->decimal('loan_amount', 10, 2);
            $table->decimal('loan_amount_term', 10, 2);
            $table->string('credit_history');
            $table->string('property_area');
            $table->string('loan_status')->nullable();
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('loans');
    }
};
