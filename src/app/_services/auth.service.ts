import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Router } from '@angular/router';

@Injectable({
  providedIn: 'root'
})
export class AuthService {

  constructor(private router:Router,private http:HttpClient) { }

  isAuthenticated():boolean{
    if (sessionStorage.getItem('token')!==null) {
        return true;
    }
    return false;
  }

  canAccess(){
    if (!this.isAuthenticated()) {
        //redirect to login
        this.router.navigate(['/login']);
    }
  }
  canAuthenticate(){
    if (this.isAuthenticated()) {
      //redirect to dashboard
      this.router.navigate(['/dashboard']);
    }
  }

  register(name:string,email:string,password:string){
      //send data to register api (firebase)
     return this.http
      .post<{idToken:string}>(
        'https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=AIzaSyB-G9QDIVOuH2CELlpy_tU7TjTk9I1S7yA',
          {displayName:name,email,password}
      );
  }

  storeToken(token:string){
      sessionStorage.setItem('token',token);
  }

  login(email:string,password:string){
    //send data to login api (firebase)
      return this.http
      .post<{idToken:string}>(
          'https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=AIzaSyB-G9QDIVOuH2CELlpy_tU7TjTk9I1S7yA',
            {email,password}
      );
  }

  detail(){
    let token = sessionStorage.getItem('token');

    return this.http.post<{users:Array<{localId:string,displayName:string}>}>(
        'https://identitytoolkit.googleapis.com/v1/accounts:lookup?key=AIzaSyB-G9QDIVOuH2CELlpy_tU7TjTk9I1S7yA',
        {idToken:token}
    );
  }

  removeToken(){
    sessionStorage.removeItem('token');
  }



}
