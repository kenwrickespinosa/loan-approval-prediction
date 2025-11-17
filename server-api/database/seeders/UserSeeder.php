<?php

namespace Database\Seeders;

use App\Models\User;
use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\Hash;

class UserSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        User::create([
            'firstname'=>'John',
            'lastname'=>'Doe',
            'Gender'=>'Male',
            'birthdate' => '2004-01-01',
            'address' => 'Philippines',
            'role' => 'staff',
            'email' => 'johndoe@example.com',
            'password' => Hash::make('johndoe123')
        ]);
    }
}
