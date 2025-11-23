<?php

namespace App\Http\Controllers;

use App\Models\Client;
use Illuminate\Http\Request;

class ClientController extends Controller
{
    public function index(Request $request) {
        $user = $request->user();

        return Client::where('user_id', $user->id)->get();
    }

    public function store(Request $request) {
        $validated = $request->validate([
            'firstname' => 'required|string|max:255',
            'lastname' => 'required|string|max:255',
            'gender' => 'required|string|max:25',
            'birthdate' => 'required|date',
            'address' => 'required|string|max:255',
            'contact_number' => 'required|string|max:25'
        ]);

        $validated['user_id'] = $request->user()->id;

        $client = Client::create($validated);

        return response()->json([
            'message' => 'Created successfully',
            'client' => $client
        ], 201);
    }
}
