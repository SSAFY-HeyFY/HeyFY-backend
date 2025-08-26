package com.ssafy.ssashinsa.heyfy.register.controller;

import com.ssafy.ssashinsa.heyfy.account.dto.AccountNoDto;
import com.ssafy.ssashinsa.heyfy.register.docs.CreateDepositAccountDocs;
import com.ssafy.ssashinsa.heyfy.register.docs.CreateForeignDepositAccountDocs;
import com.ssafy.ssashinsa.heyfy.register.docs.RegisterAccountDocs;
import com.ssafy.ssashinsa.heyfy.register.docs.RegisterForeignAccountDocs;
import com.ssafy.ssashinsa.heyfy.register.dto.AccountCreationResponseDto;
import com.ssafy.ssashinsa.heyfy.register.service.RegisterService;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.create.ShinhanCreateDepositResponseDto;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Tag(name = "계좌 생성&등록", description = "신한은행 계좌 생성 및 등록 관련 API")
@RestController
@RequestMapping("/register")
@RequiredArgsConstructor
public class RegisterController {

    private final RegisterService registerService;

    @CreateDepositAccountDocs
    @PostMapping("/createdeposit")
    public ResponseEntity<AccountCreationResponseDto> createDepositAccount() {

        ShinhanCreateDepositResponseDto response = registerService.createDepositAccount();

        String message = response.getHeader().getResponseMessage();
        String accountNo = response.getREC().getAccountNo();
        String currency = response.getREC().getCurrency().getCurrency();

        AccountCreationResponseDto simplifiedResponse = new AccountCreationResponseDto(message, accountNo, currency);

        return ResponseEntity.ok(simplifiedResponse);
    }

    @CreateForeignDepositAccountDocs
    @PostMapping("/createforeigndeposit")
    public ResponseEntity<AccountCreationResponseDto> createDepositForeignAccount() {

        ShinhanCreateDepositResponseDto response = registerService.createForeignDepositAccount();

        String message = response.getHeader().getResponseMessage();
        String accountNo = response.getREC().getAccountNo();
        String currency = response.getREC().getCurrency().getCurrency();

        AccountCreationResponseDto simplifiedResponse = new AccountCreationResponseDto(message, accountNo, currency);

        return ResponseEntity.ok(simplifiedResponse);
    }

    @RegisterAccountDocs
    @PostMapping("/registeraccount")
    public ResponseEntity<AccountNoDto> registerAccount(@RequestBody AccountNoDto accountNoDto) {

        String accountNo = accountNoDto.getAccountNo();
        String newAccountNo = registerService.registerAccountFromShinhan(accountNo);
        AccountNoDto responseDto = new AccountNoDto(newAccountNo);

        return ResponseEntity.ok(responseDto);
    }

    @RegisterForeignAccountDocs
    @PostMapping("/registerforeignaccount")
    public ResponseEntity<AccountNoDto> registerForeignAccount(@RequestBody AccountNoDto accountNoDto) {

        String accountNo = accountNoDto.getAccountNo();
        String newAccountNo = registerService.registerForeignAccountFromShinhan(accountNo);
        AccountNoDto responseDto = new AccountNoDto(newAccountNo);

        return ResponseEntity.ok(responseDto);
    }

}
