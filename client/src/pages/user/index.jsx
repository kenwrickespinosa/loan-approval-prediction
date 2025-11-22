import React, { useState } from 'react'
import UserForm from './UserForm'

function index() {
  const [isUserForm, setIsUserForm] = useState(false);

  return (
    <div>
      <Button type="button" onClick={handleUserFormOpen}>Create User</Button>
    </div>
  )
}

export default index