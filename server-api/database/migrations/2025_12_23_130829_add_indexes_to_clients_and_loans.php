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
        Schema::table('clients', function (Blueprint $table) {
            $table->index('gender');
        });

        Schema::table('loans', function (Blueprint $table) {
            $table->index('married');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('clients', function (Blueprint $table) {
            $table->dropIndex('clients_gender_index');
        });

        Schema::table('loans', function (Blueprint $table) {
            $table->dropIndex('loans_married_index');
        });
    }
};
