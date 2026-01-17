<?php

namespace App\Http\Controllers;

use App\Models\User;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Hash;

class AuthController extends Controller
{
    public function login(Request $request)
    {
        $validated = $request->validate([
            'email' => 'required|string|email',
            'password' => 'required|min:8'
        ]);

        $user = User::where('email', $validated['email'])->first();

        if (!$user || !Hash::check($validated['password'], $user->password)) {
            return response()->json([
                'message' => 'Invalid credentials'
            ], 401);
        }

        $token = $user->createToken('auth_token')->plainTextToken;

        return response()->json([
            'message' => 'Login successfully',
            'token' => $token,
            'token_type' => 'Bearer',
            'user' => $user
        ]);
    }

    public function register(Request $request)
    {
        $validated = $request->validate([
            'firstname' => 'required|string|max:255',
            'lastname' => 'required|string|max:255',
            'gender' => 'required|string|in:Male,Female,Other',
            'birthdate' => 'required|date|before:today',
            'address' => 'required|string|max:255',
            'role' => 'required|string|in:staff',
            'email' => 'required|string|email',
            'password' => 'required|string|min:8'
        ]);

        $user = User::create([
            'firstname' => $validated['firstname'],
            'lastname' => $validated['lastname'],
            'gender' => $validated['gender'],
            'birthdate' => $validated['birthdate'],
            'address' => $validated['address'],
            'role' => $validated['role'] ?? 'staff',
            'email' => $validated['email'],
            'password' => Hash::make($validated['password']),
        ]);

        $token = $user->createToken('token')->plainTextToken;

        return response()->json([
            'user' => $user,
            'token' => $token,
            'message' => 'User registered successfully',
        ]);
    }

    public function logout(Request $request)
    {
        $request->user()->currentAccessToken()->delete();
        return response()->json(['message' => 'Logged out']);
    }
}
